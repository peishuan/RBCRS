import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

class UserEncoder(nn.Module):
    def __init__(self, layer_sizes, n_movies, f):
        """
        :param layer_sizes: list giving the size of each layer
        :param n_movies:
        :param f:
        """
        super(UserEncoder, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(in_features=n_movies, out_features=layer_sizes[0]) if i == 0
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
                input = layer(input)
            else:
                input = self.f(layer(input))
        return input


class AutoRec(nn.Module):
    """
    User-based Autoencoder for Collaborative Filtering
    """

    def __init__(self,
                 n_movies,
                 params,
                 resume=None):
        super(AutoRec, self).__init__()
        self.params = params
        self.cuda_available = torch.cuda.is_available()
        self.n_movies = n_movies
        self.layer_sizes = params['layer_sizes']
        if params['g'] == 'identity':
            self.g = lambda x: x
        elif params['g'] == 'sigmoid':
            self.g = nn.Sigmoid()
        elif params['g'] == 'relu':
            self.g = nn.ReLU()
        else:
            raise ValueError("Got invalid function name for g : {}".format(params['g']))

        if resume is not None:
            # Load pretrained model, keeping only the first n_movies. (see merge_indexes function in match_movies)
            if self.cuda_available:
                checkpoint = torch.load(resume)
            else:
                checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)


            # get layer sizes from checkpoint if saved
            if "layer_sizes" in checkpoint:
                self.layer_sizes = checkpoint["layer_sizes"]
            self.encoder = UserEncoder(layer_sizes=self.layer_sizes, n_movies=n_movies, f=params['f'])
            self.user_representation_size = self.layer_sizes[-1]
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
            self.encoder = UserEncoder(layer_sizes=self.layer_sizes, n_movies=n_movies, f=params['f'])
            self.user_representation_size = self.layer_sizes[-1]
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
        
            #沒有要backward的話使用能節省一點內存
            
        self.eval()
        batch_loader.batch_index[subset] = 0
        n_batches = batch_loader.n_batches[subset]

        losses = []
        #with torch.no_grad():
        for _ in tqdm(range(n_batches)):
            # load batch
            batch = batch_loader.load_batch(subset=subset, batch_input=batch_input)
            if self.cuda_available:
                batch["input"] = batch["input"].cuda()
                batch["target"] = batch["target"].cuda()
            # compute output and loss
            output = self.forward(batch["input"])
            loss = criterion(output, batch["target"])
            # print("in loop: each loss")
            for val in loss.data:
                losses.append(val)
        # normalize loss and reset nb of ratings observed
        final_loss = criterion.normalize_loss_reset(np.sum(losses))
        # print("{} loss : {}".format(subset, torch.mean(final_loss))
        print("{} loss with input={} : {}".format(subset, batch_input, final_loss))
        self.train()
        return  final_loss
    
    def test(self, batch_loader, subset):
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
        avg_acc1 = []
        avg_acc5 = []
        avg_acc10 = []
        avg_acc50 = []
        avg_acc100 = []
        avg_rec1 = []
        avg_rec5 = []
        avg_rec10 = []
        avg_rec50 = []
        avg_rec100 = []
        avg_pre1 = []
        avg_pre5 = []
        avg_pre10 = []
        avg_pre50 = []
        avg_pre100 = []
        with torch.no_grad():
            for _ in tqdm(range(n_batches)):
                # load batch
                batch = batch_loader.load_batch(subset=subset)
                if self.cuda_available:
                    batch["input_dist"] = batch["input_dist"].cuda()
                    batch["target"] = batch["target"].cuda()
                # compute output and loss
                output = self.forward(batch["input_dist"])
                # PREDICTIONS
                # pred = np.round(output.cpu().data.numpy())
                # extract target from batch
                target = []
                for val in batch["target"].cpu().data.numpy():
                    for n in val:
                        target.append(n)
                
                # # # # build prediction list for one to one
                #input_idx = []
                #for i in (torch.topk(batch["input_dist"],1).indices).cpu().data.numpy():
                #    for val in i.tolist():
                #      input_idx.append(val)
                ## top 1
                #pred_1 = [0] * 6924
                #for i in (torch.topk(output,1).indices).cpu().data.numpy():
                #    if(i in input_idx):
                #        pred_1[(torch.topk(output,2).indices[0][1]).cpu().data.numpy()] = 1
                #    else:
                #        pred_1[int(i)] = 1
                
                ## top 5
                #pred_5 = [0] * 6924
                #for i in (torch.topk(output,5).indices).cpu().data.numpy():
                #    if(input_idx in i):
                #        # 將output中input的index去掉 增加對上target的機會
                #        for val in i:
                #            if(val == input_idx):
                #                continue
                #        else:
                #            pred_5[int(val)] = 1
                #        pred_5[int(torch.topk(output,6).indices[0][5])] = 1
                #    else:
                #        for val in i:
                #            pred_5[int(val)] = 1

                
                # top 10
                #pred_10 = [0] * 6924
                #for i in (torch.topk(output,10).indices).cpu().data.numpy():
                #    if(input_idx in i):
                #        # 將output中input的index去掉 增加對上target的機會
                #        for val in i:
                #            if(val == input_idx):
                #                continue
                #        else:
                #            pred_10[int(val)] = 1
                #        pred_10[int(torch.topk(output,11).indices[0][10])] = 1
                #    else:
                #        for val in i:
                #            pred_10[int(val)] = 1
                
                # top 50
                #pred_50 = [0] * 6924
                #for i in (torch.topk(output,50).indices).cpu().data.numpy():
                #    if(input_idx in i):
                #        # 將output中input的index去掉 增加對上target的機會
                #        for val in i:
                #            if(val == input_idx):
                #                continue
                #        else:
                #            pred_50[int(val)] = 1
                #        pred_50[int(torch.topk(output,51).indices[0][50])] = 1
                #    else:
                #        for val in i:
                #            pred_50[int(val)] = 1

                # top 100
                #pred_100 = [0] * 6924
                #for i in (torch.topk(output,100).indices).cpu().data.numpy():
                #    if(input_idx in i):
                #        # 將output中input的index去掉 增加對上target的機會
                #        for val in i:
                #            if(val == input_idx):
                #                continue
                #        else:
                #            pred_100[int(val)] = 1
                #        pred_100[int(torch.topk(output,101).indices[0][100])] = 1
                #    else:
                #        for val in i:
                #            pred_100[int(val)] = 1

                # # build prediction list for two to one
                input_idx = []
                for i in (torch.topk(batch["input_dist"],2).indices).cpu().data.numpy():
                    for val in i.tolist():
                      input_idx.append(val)
                # top 1
                pred_1 = [0] * 6924
                for i in (torch.topk(output,1).indices).cpu().data.numpy():
                    if(int(i) in input_idx):
                        pred_1[(torch.topk(output,2).indices[0][1]).cpu().data.numpy()] = 1
                    else:
                        pred_1[int(i)] = 1
                
                # top 5
                pred_5 = [0] * 6924
                dup = 0
                for i in (torch.topk(output,5).indices).cpu().data.numpy():
                    if(input_idx in i):
                        # 將output中input的index去掉 增加對上target的機會
                        for val in i:
                            if(val == input_idx):
                                dup = dup +1
                                continue
                        else:
                            pred_5[int(val)] = 1

                        if dup > 1: 
                            for i in range(dup):
                                pred_5[int(torch.topk(output,6+i).indices[0][5+i])] = 1
                    else:
                        for val in i:
                            pred_5[int(val)] = 1

                
                # # top 10
                pred_10 = [0] * 6924
                dup = 0
                for i in (torch.topk(output,10).indices).cpu().data.numpy():
                    if(input_idx in i):
                        # 將output中input的index去掉 增加對上target的機會
                        for val in i:
                            if(val == input_idx):
                                dup = dup +1
                                continue
                        else:
                            pred_10[int(val)] = 1
                            
                        if dup > 1: 
                            for i in range(dup):
                                pred_10[int(torch.topk(output,11+i).indices[0][10+i])] = 1
                    else:
                        for val in i:
                            pred_10[int(val)] = 1
                
                # # top 50
                pred_50 = [0] * 6924
                dup = 0
                for i in (torch.topk(output,50).indices).cpu().data.numpy():
                    if(input_idx in i):
                        # 將output中input的index去掉 增加對上target的機會
                        for val in i:
                            if(val == input_idx):
                                dup = dup +1
                                continue
                        else:
                            pred_50[int(val)] = 1
                            
                        if dup > 1: 
                            for i in range(dup):
                                pred_50[int(torch.topk(output,51+i).indices[0][50+i])] = 1
                    else:
                        for val in i:
                            pred_50[int(val)] = 1

                # # top 100
                pred_100 = [0] * 6924
                dup = 0
                for i in (torch.topk(output,100).indices).cpu().data.numpy():
                    if(input_idx in i):
                        # 將output中input的index去掉 增加對上target的機會
                        for val in i:
                            if(val == input_idx):
                                dup = dup +1
                                continue
                        else:
                            pred_100[int(val)] = 1
                            
                        if dup > 1: 
                            for i in range(dup):
                                pred_100[int(torch.topk(output,101+i).indices[0][100+i])] = 1
                    else:
                        for val in i:
                            pred_100[int(val)] = 1


                avg_acc1.append(accuracy_score(target,pred_1))
                avg_acc5.append(accuracy_score(target,pred_5))
                avg_acc10.append(accuracy_score(target,pred_10))
                avg_acc50.append(accuracy_score(target,pred_50))
                avg_acc100.append(accuracy_score(target,pred_100))
                avg_rec1.append(recall_score(target,pred_1))
                avg_rec5.append(recall_score(target,pred_5))
                avg_rec10.append(recall_score(target,pred_10))
                avg_rec50.append(recall_score(target,pred_50))
                avg_rec100.append(recall_score(target,pred_100))
                avg_pre1.append(precision_score(target,pred_1))
                avg_pre5.append(precision_score(target,pred_5))
                avg_pre10.append(precision_score(target,pred_10))
                avg_pre50.append(precision_score(target,pred_50))
                avg_pre100.append(precision_score(target,pred_100))
        
        print("Accuracy @1 on test set is" , sum(avg_acc1) / len(avg_acc1))
        print("Precision @1 on test set is" , sum(avg_pre1) / len(avg_pre1))
        print("Recall @1 on test set is" , sum(avg_rec1) / len(avg_rec1))

        print("\nAccuracy @5 on test set is" , sum(avg_acc5) / len(avg_acc5))
        print("Precision @5 on test set is" , sum(avg_pre5) / len(avg_pre5))
        print("Recall @5 on test set is" , sum(avg_rec5) / len(avg_rec5))

        print("\nAccuracy @10 on test set is" , sum(avg_acc10) / len(avg_acc10))
        print("Precision @10 on test set is" , sum(avg_pre10) / len(avg_pre10))
        print("Recall @10 on test set is" , sum(avg_rec10) / len(avg_rec10))

        print("\nAccuracy @50 on test set is" , sum(avg_acc50) / len(avg_acc50))
        print("Precision @50 on test set is" , sum(avg_pre50) / len(avg_pre50))
        print("Recall @50 on test set is" , sum(avg_rec50) / len(avg_rec50))

        print("\nAccuracy @100 on test set is" , sum(avg_acc100) / len(avg_acc100))
        print("Precision @100 on test set is" , sum(avg_pre100) / len(avg_pre100))
        print("Recall @100 on test set is" , sum(avg_rec100) / len(avg_rec100))


class ReconstructionLoss(nn.Module):
    def __init__(self):
        super(ReconstructionLoss, self).__init__()
        # if torch.cuda.is_available():
        #     self.cuda()
        self.nb_observed_targets = 0

    def forward(self, model_output, target):
        # only consider the observed targets
        mask = (target != -1)
        observed_input = torch.masked_select(model_output, mask)
        observed_target = torch.masked_select(target, mask)
        # print(observed_input)
        # print(observed_target)
        # print(observed_input.shape)
        # print(observed_target.shape)
        loss = nn.BCEWithLogitsLoss(size_average=False, reduce=False)
        # print("model output:{}".format(model_output.size()))
        # print("target:{}".format(target.size()))

        self.nb_observed_targets += len(observed_target)
        output_loss = loss(observed_input, observed_target.float())
        # sum_loss = 0
        # for l in output_loss:
        #     sum_loss = sum_loss+l
        # print("output loss:{}".format(output_loss.size()))
        return output_loss

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


