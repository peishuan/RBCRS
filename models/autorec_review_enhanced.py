import os
import config
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sklearn.metrics
from tqdm import tqdm
from models.hierarchical_review_rnn import RHRNN
from models.review_rep_trainer import ReviewRepTrainer

class UserEncoder(nn.Module):
    def __init__(self, layer_sizes, in_features, f):
        """
        :param layer_sizes: list giving the size of each layer
        :param n_movies:
        :param f:
        """
        super(UserEncoder, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(in_features=in_features, out_features=layer_sizes[0]) if i == 0
                                     else nn.Linear(in_features=layer_sizes[i - 1], out_features=layer_sizes[i])
                                     for i in range(len(layer_sizes))])
        if f == 'identity':
            self.f = lambda x: x
        elif f == 'sigmoid':
            self.f = nn.Sigmoid()
        elif f == 'relu':
            self.f = nn.ReLU()
        else:
            raise ValueError("Got invalid function name for f : {}".format(f))

    def forward(self, input, raw_last_layer=False):
        for (i, layer) in enumerate(self.layers):
            if raw_last_layer and i == len(self.layers) - 1:
                # do not apply activation to last layer
                input = layer(input.float())
            else:
                input = self.f(layer(input))
        return input

class R_AutoRec(nn.Module):
    

    def __init__(self,
                train_vocab,
                 n_movies,
                 params,
                 resume=None):
        super(R_AutoRec, self).__init__()
        self.params = params
        self.cuda_available = torch.cuda.is_available()
        self.n_movies = n_movies

        if resume is not None:
            # Load pretrained model, keeping only the first n_movies. (see merge_indexes function in match_movies)
            if self.cuda_available:
                checkpoint = torch.load(resume)
            else:
                checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)
            # get layer sizes from checkpoint if saved
            if "layer_sizes" in checkpoint:
                self.layer_sizes = checkpoint["layer_sizes"]
            
            self.review_encoder = ReviewRepTrainer(train_vocab,params['review_rep_params'],self.layer_sizes,f=self.f, n_movies=self.n_movies, train_gensen=False, gensen=True, conv_bidrectional=False)
            self.rev_encoder_output_size = self.params['review_rep_params']['conversation_encoder_hidden_size']
            self.encoder = UserEncoder(layer_sizes=self.params['review_rep_params']['layer_sizes'], in_features=self.rev_encoder_output_size, f=params['f'])
            self.user_representation_size = self.params['review_rep_params']['layer_sizes'][-1]
            self.decoder = nn.Linear(in_features=self.user_representation_size, out_features=n_movies)
            model_dict = self.state_dict()
            # load all weights except for the weights of the first layer and the decoder
            model_dict.update({k: v for k, v in checkpoint['state_dict'].items()
                               if k != "encoder.layers.0.weight" and "decoder" not in k})
            # load first layer and decoder: assume the movies to keep are the n_movies first movies
            encoder0weight = checkpoint["state_dict"]["encoder.layers.0.weight"][:, :self.n_movies]
            decoderweight = checkpoint["state_dict"]["decoder.weight"][:self.n_movies, :]
            decoderbias = checkpoint["state_dict"]["decoder.bias"][:self.n_movies]
            # If checkpoint has fewer movies than the model, append zeros (no actual recommendations for new movies)
            # (When using an updated movie list)
            if encoder0weight.shape[1] < self.n_movies:
                tt = torch.cuda.FloatTensor if self.cuda_available else torch.FloatTensor
                encoder0weight = torch.cat((
                    encoder0weight,
                    torch.zeros(encoder0weight.shape[0], self.n_movies - encoder0weight.shape[1], out=tt())), dim=1)
                decoderweight = torch.cat((
                    decoderweight,
                    torch.zeros(self.n_movies - decoderweight.shape[0], decoderweight.shape[1], out=tt())), dim=0)
                decoderbias = torch.cat((
                    decoderbias, torch.zeros(self.n_movies - decoderbias.shape[0], out=tt())), dim=0)
            model_dict.update({
                "encoder.layers.0.weight": encoder0weight,
                "decoder.weight": decoderweight,
                "decoder.bias": decoderbias,
            })
            self.load_state_dict(model_dict)
        else:
            self.review_encoder = ReviewRepTrainer(train_vocab,self.params['review_rep_params'], n_movies=self.n_movies, train_gensen=False, gensen=True, conv_bidrectional=False)
            self.encoder_output_size = self.params['review_rep_params']['conversation_encoder_hidden_size']
            self.encoder = UserEncoder(layer_sizes=self.params['review_rep_params']['layer_sizes'], in_features=self.encoder_output_size, f=params['review_rep_params']['f'])
            self.user_representation_size = self.params['review_rep_params']['layer_sizes'][-1]
            self.decoder = nn.Linear(in_features=self.user_representation_size, out_features=n_movies)

        if self.cuda_available:
            self.cuda()

    def forward(self, input, additional_context=None, range01=True):
        """

        :param input: (batch, n_movies)
        :param additional_context: potential information to add to user representation (batch, user_rep_size)
        :param range01: If true, apply sigmoid to the output
        :return: output recommendations (batch, n_movies)
        """
        # get user representation
        print(input)
        encoded = self.encoder(input, raw_last_layer=True)
        # eventually use additional context
        if additional_context is not None:
            encoded = self.encoder.f(encoded + additional_context)
        else:
            encoded = self.encoder.f(encoded)
        # decode to get movie recommendations
        if range01:
            return self.g(self.decoder(encoded))
        else:
            return self.decoder(encoded)


    def evaluate(self, batch_loader, criterion, subset, batch_input):
        """
        Evaluate function
        :param batch_loader:
        :param criterion:
        :param batch_input:
        :param subset: in {"test", "valid", "train"}. Susbet on which to evaluate
        :return: the mean loss.
        """
        self.eval()
        batch_loader.batch_index[subset] = 0
        n_batches = batch_loader.n_batches[subset]

        losses = []
        for _ in tqdm(range(n_batches)):
            # load batch
            batch = batch_loader.load_batch(subset=subset, batch_input=batch_input)
            if self.cuda_available:
                batch["reviews"] = batch["reviews"].cuda()
                batch["target"] = batch["target"].cuda()
            # compute output and loss
            # output = batch["target"].cpu().data.numpy()
            output = self.forward(batch["reviews"])
            loss = criterion(output, batch["target"])
            losses.append(loss.data)
            # losses.append(loss.data[0])
        # normalize loss and reset nb of ratings observed
        final_loss = criterion.normalize_loss_reset(np.sum(losses))
        print("{} loss with input={} : {}".format(subset, batch_input, final_loss))
        self.train()
        return final_loss


class ReconstructionLoss(nn.Module):
    def __init__(self):
        super(ReconstructionLoss, self).__init__()
        # Sum of losses
        self.mse_loss = nn.MSELoss(size_average=False)
        # Keep track of number of observer targets to normalize loss
        self.nb_observed_targets = 0

    def forward(self, input, target):
        # only consider the observed targets
        mask = (target != -1)
        print(target)
        observed_input = torch.masked_select(input, mask)
        observed_target = torch.masked_select(target, mask)
        # increment nb of observed targets
        self.nb_observed_targets += len(observed_target)
        loss = self.mse_loss(observed_input, observed_target)
        return loss

    def normalize_loss_reset(self, loss):
        """
        returns loss divided by nb of observed targets, and reset nb_observed_targets to 0
        :param loss: Total summed loss
        :return: mean loss
        """
        if self.nb_observed_targets == 0:
            raise ValueError(
                "Nb observed targets was 0. Please evaluate some examples before calling normalize_loss_reset")
        n_loss = loss / self.nb_observed_targets
        self.nb_observed_targets = 0
        return n_loss
