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
            return (self.Iliked(conversation_representations))
        else:
            conversation_representations = self.encoder(input, return_all=False)
            return (self.Iliked(conversation_representations))
    
    def return_rep(self, input, return_liked_probability=False):
        """
        :param input: dictionary that contains the different inputs: dialogue, senders, movie_occurrences
        :return:
        """
        if return_liked_probability:
            conversation_representations = self.encoder(input, return_all=True)
            return conversation_representations
        else:
            conversation_representations = self.encoder(input, return_all=False)
            return conversation_representations
    
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
            Iliked = torch.max(output, 1)[1].squeeze().cpu()

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

                #PREDICTIONS
                # pred = np.round(output.cpu().data.numpy())
                #extract target from batch
                target = []
                for val in batch["target"].cpu().data.numpy():
                    for n in val:
                        target.append(n)
                
                # build prediction list
                # top 1
                pred_1 = [0] * 6924
                for i in (torch.topk(output,1).indices).cpu().data.numpy():
                    pred_1[int(i)] = 1
                
                # top 5
                pred_5 = [0] * 6924
                for i in (torch.topk(output,5).indices).cpu().data.numpy():
                    for val in i:
                        pred_5[int(val)] = 1
                
                # top 10
                pred_10 = [0] * 6924
                for i in (torch.topk(output,10).indices).cpu().data.numpy():
                    for val in i:
                        pred_10[int(val)] = 1
                
                # top 50
                pred_50 = [0] * 6924
                for i in (torch.topk(output,50).indices).cpu().data.numpy():
                    for val in i:
                        pred_50[int(val)] = 1

                # top 100
                pred_100 = [0] * 6924
                for i in (torch.topk(output,100).indices).cpu().data.numpy():
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
            #result.append({'movie_name' : batch['movieName'][0][0] , 'rep' :output.cpu().detach().numpy()} , ignore_index=True)

            #print(output.size())
            #print(output.cpu().detach().numpy())
        return result
        

            


class ReconstructionLoss(nn.Module):
    def __init__(self):
        super(ReconstructionLoss, self).__init__()
        if torch.cuda.is_available():
            self.cuda()

    def forward(self, model_output, target):
        loss = nn.BCEWithLogitsLoss(size_average=False, reduce=False)
        # print("model output:{}".format(model_output.size()))
        # print("target:{}".format(target.size()))
        
        output_loss = loss(model_output, target.float())
        # sum_loss = 0
        # for l in output_loss:
        #     sum_loss = sum_loss+l
        # print("output loss:{}".format(output_loss.size()))
        return output_loss
