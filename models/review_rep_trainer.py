import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sklearn.metrics
from tqdm import tqdm
import pandas as pd
from models.hierarchical_review_rnn import RHRNN
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

class ReviewRepTrainer(nn.Module):
    def __init__(self,train_vocab,params,n_movies,train_gensen=False, gensen=True, conv_bidrectional=False,resume=None):

        """
        :param layer_sizes: list giving the size of each layer
        :param n_movies:
        :param f:
        """
        super(ReviewRepTrainer, self).__init__()
        self.params = params
        self.cuda_available = torch.cuda.is_available()
        self.train_gensen = train_gensen
        self.n_movies = n_movies

        torch.backends.cudnn.enabled = False 

        if resume is not None:
            if self.cuda_available:
                checkpoint = torch.load(resume)
            else:
                checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)
        self.encoder = RHRNN(
            params=self.params,
            train_vocabulary=train_vocab,
            gensen=gensen,
            train_gensen=self.train_gensen,
            conv_bidirectional=conv_bidrectional,
        )
        encoder_output_size = self.params['conversation_encoder_hidden_size']

        self.Iliked = nn.Linear((1 + conv_bidrectional) * encoder_output_size, n_movies)
        
        if self.cuda_available:
            self.cuda()
        
        if resume is not None:
            # load weights from saved model
            self.load_state_dict(checkpoint['state_dict'])

    def forward(self, input, return_liked_probability=False):
        """
        :param input: dictionary that contains the different inputs: dialogue, senders, movie_occurrences
        :return:
        """
        # torch.backends.cudnn.enabled = False 
        if return_liked_probability:
            conversation_representations = self.encoder(input, return_all=True)
            return (self.Iliked(torch.tanh(conversation_representations)))
        else:
            conversation_representations = self.encoder(input, return_all=False)
            return (self.Iliked(torch.tanh(conversation_representations)))
    
    def return_rep(self, input, return_liked_probability=False):
        """
        :param input: dictionary that contains the different inputs: dialogue, senders, movie_occurrences
        :return:
        """
        if return_liked_probability:
            conversation_representations = self.encoder(input, return_all=True)
            return torch.tanh(conversation_representations)
        else:
            conversation_representations = self.encoder(input, return_all=False)
            return torch.tanh(conversation_representations)
    
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
        # torch.backends.cudnn.enabled = False 
        total = 0
        correct = 0
        losses = []
        matrix_size = 18
        Iconfusion_matrix = np.zeros((matrix_size, matrix_size), dtype=int)
        wrongs = 0
        wrongs_with_disagreement = 0
        for i in tqdm(range(n_batches)):
            # load batch
            batch = batch_loader.load_batch(subset)
            if self.cuda_available:
                batch["reviews"] = batch["reviews"].cuda()
                batch["target"] = batch["target"].cuda()
            # compute output and loss
            
            target = batch["target"].cpu().data.numpy()
            output = self.forward(batch)
            loss = criterion(output, batch["target"])
            losses.append(loss.data)

            # get the arg max for the categorical output
            #Iliked = torch.max(output, 1)[1].squeeze().cpu()

            # increment number of wrong predictions (either seeker and recommender)
            # wrongs += np.sum(1 * (Iliked.data.numpy() != target) )
           
        # print("{} wrong answers for liked label, for {} of those there was a disagreement between workers"
        #       .format(wrongs, wrongs_with_disagreement))
        print("{} loss : {}".format(subset, torch.mean(torch.stack(losses))))
        self.train()
        return torch.mean(torch.stack(losses))

    def test(self, batch_loader, subset):
        self.eval()
        batch_loader.batch_index[subset] = 0
        n_batches = batch_loader.n_batches[subset]

        test_loss = 0
        correct = 0
        avg_hit1 = []
        avg_hit3 = []
        avg_hit5 = []
        avg_mrr1 = []
        avg_mrr3 = []
        avg_mrr5 = []
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
        with torch.no_grad(): # disable gradient calculation for efficiency
            for i in tqdm(range(n_batches)):
            # for data, target in test_loader:
                # load batch
                batch = batch_loader.load_batch(subset)
                if self.cuda_available:
                    batch["reviews"] = batch["reviews"].cuda()
                    batch["target"] = batch["target"].cuda()
                # Prediction [1,6924]
                output = self.forward(batch)

                target = []
                for val in batch["target"].cpu().data.numpy():
                    for n in val:
                        target.append(n)
                ans = [x for x, z in enumerate(target) if z ==1]
                
                hit1= [] 
                mrr1 = []
                topk1 = []
                for i in (torch.topk(output,1).indices).cpu().data.numpy().tolist():
                    for j in i:
                        topk1.append(j)
                for a in ans:
                    if a in topk1:
                        hit1.append(1)
                        mrr1.append(1)
                    else:
                        hit1.append(0)
                        mrr1.append(0)
                avg_hit1.append(sum(hit1) / len(hit1))
                avg_mrr1.append(sum(mrr1) / len(mrr1))

                hit3= [] 
                mrr3 = []
                topk3 = []
                for i in (torch.topk(output,3).indices).cpu().data.numpy().tolist():
                    for j in i:
                        topk3.append(j)
                for a in ans:   
                    if a in topk3:
                        hit3.append(1)
                        mrr3.append(1/(1+topk3.index(a)))
                    else:
                        hit3.append(0)
                        mrr3.append(0)
                if(any(ele == 1 for ele in hit3)):
                    avg_hit3.append(1)
                else:
                    avg_hit3.append(0)

                avg_mrr3.append(max(mrr3))

                hit5= [] 
                mrr5 = []
                topk5 = []
                for i in (torch.topk(output,5).indices).cpu().data.numpy().tolist():
                    for j in i:
                        topk5.append(j)
                for a in ans:
                    if a in topk5:
                        hit5.append(1)
                        mrr5.append(1/(1+topk5.index(a)))
                    else:
                        hit5.append(0)
                        mrr5.append(0)
                if(any(ele == 1 for ele in hit5)):
                    avg_hit5.append(1)
                else:
                    avg_hit5.append(0)
                avg_mrr5.append(max(mrr5))

        print("HR @1 on test set is" , sum(avg_hit1) / len(avg_hit1))
        print("HR @3 on test set is" , sum(avg_hit3) / len(avg_hit3))
        print("HR @5 on test set is" , sum(avg_hit5) / len(avg_hit5))

        print("MRR @1 on test set is" , sum(avg_mrr1) / len(avg_mrr1))
        print("MRR @3 on test set is" , sum(avg_mrr3) / len(avg_mrr3))
        print("MRR @5 on test set is" , sum(avg_mrr5) / len(avg_mrr5))

    def build_rep_evaluate(self, batch_loader,  subset, batch_input):
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

        total = 0
        correct = 0
        losses = []
        matrix_size = 18
        Iconfusion_matrix = np.zeros((matrix_size, matrix_size), dtype=int)
        wrongs = 0
        wrongs_with_disagreement = 0

        
        result = {}
        for i in tqdm(range(n_batches)):
            # load batch
            batch = batch_loader.load_batch(subset)
            if self.cuda_available:
                batch["reviews"] = batch["reviews"].cuda()
                batch["target"] = batch["target"].cuda()
            # compute output and loss
            target = batch["target"].cpu().data.numpy()
            output = self.return_rep(batch)
            result[batch['movieName'][0][0]] = output.cpu().detach().numpy()
            #print(result[batch['movieName'][0][0]].shape)
            #print(output.cpu().detach().numpy())
            #result.append({'movie_name' : batch['movieName'][0][0] , 'rep' :output.cpu().detach().numpy()} , ignore_index=True)

            #print(output.cpu().detach().numpy())
        return result
        

            


class ReconstructionLoss(nn.Module):
    def __init__(self):
        super(ReconstructionLoss, self).__init__()
        if torch.cuda.is_available():
            self.cuda()

    def forward(self, model_output, target):
        loss = nn.BCEWithLogitsLoss(size_average=False, reduce=False)
        output_loss = loss(model_output, target.float())

        #loss = nn.NLLLoss()

        #trandform from bce loss
        #target_input = []
        #for val in target.cpu().data.numpy():
        #    not0 = [x for x, z in enumerate(val) if z ==1]
        #    target_input.append(not0[0])

        #ans = torch.LongTensor(target_input).cuda()
        #output_loss = loss(model_output,ans)
        
        return output_loss
