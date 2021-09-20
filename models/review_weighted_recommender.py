import os
import config
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import operator
import sklearn.metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from tqdm import tqdm
from models.sentiment_analysis import SentimentAnalysis
from numpy import newaxis
from torch.autograd import Variable
from sklearn.metrics import classification_report

class Review_weighted_recommender(nn.Module):
    """
    Recommender system that takes a dialogue as input. Runs sentiment analysis on all mentioned movies,
    then uses autorec to provide movie recommendations at each stage of the conversation
    """
    def __init__(self,
                 movie_rep_matrix,
                 n_movies,
                 params,
                 cuda=None,
                 gensen=True,
                 resume=None ):
        super(Review_weighted_recommender, self).__init__()
        self.params = params
        self.n_movies = n_movies
        self.movie_rep_matrix = movie_rep_matrix

        self.user_representation_size =self.params['conversation_encoder_hidden_size']
        self.cuda_available = torch.cuda.is_available()
        if resume is not None:
            if self.cuda_available:
                checkpoint = torch.load(resume)
            else:
                checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)

        if cuda is None:
            self.cuda_available = torch.cuda.is_available()
        else:
            self.cuda_available = cuda

        encoder_output_size = self.params['conversation_encoder_hidden_size']

        #trainable movie rep
        self.rep_emb = nn.Linear(n_movies, encoder_output_size,bias=False)
        #initial
        #self.rep_emb.weight.data = torch.transpose(nn.init.xavier_uniform_(torch.DoubleTensor(6924,512),gain=1), 0, 1)
        #

        #emb pretrain
        self.rep_emb.weight.data= torch.transpose(torch.DoubleTensor(self.movie_rep_matrix.squeeze()), 0, 1)
        ##set to learnable or not
        for param in self.rep_emb.parameters():
            ##fixed
            #param.requires_grad = False
            ##learnable
            param.requires_grad = True
        #trainable movie rep end

        self.liked_movies_pred = nn.Linear(encoder_output_size,n_movies)

        if self.cuda_available:
            self.cuda()

        if resume is not None:
            # load weights from saved model
            self.load_state_dict(checkpoint['state_dict'])

    def forward(self, input):
        #for training data
        #weighted_movie = self.rep_emb(input['input_dist'])
        ##weighted_movie = self.rep_emb(input)

        # # for concating the components
        ##weighted_movie = operator.matmul(input.cpu().numpy(),self.movie_rep_matrix.squeeze())
        input = input.type(torch.DoubleTensor)
        weighted_movie = self.rep_emb(input.cuda())
        #weighted_movie = self.rep_emb(input)
        

        if self.cuda_available:
            #output = self.liked_movies_pred(Variable(torch.from_numpy(weighted_movie)).float().cuda())
            output = self.liked_movies_pred((weighted_movie).float().cuda())
        else:
            output = self.liked_movies_pred(weighted_movie.float())

        #for recommending item, use softmax
        #output = torch.softmax(output,dim=1)
 
        return output
    
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
                batch["input_dist"] = batch["input_dist"].cuda()
                batch["target"] = batch["target"].cuda()
            # compute output and loss
            target = batch["target"].cpu().data.numpy()
            output = self.forward(batch)
            loss = criterion(output, batch["target"])
            losses.append(loss.data)

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
        avg_hit10 = []
        avg_mrr1 = []
        avg_mrr3 = []
        avg_mrr5 = []
        avg_mrr10 = []
        sum_ans = []
        counter = 0

        with torch.no_grad(): # disable gradient calculation for efficiency
            for i in tqdm(range(n_batches)):
            # for data, target in test_loader:
                # load batch
                batch = batch_loader.load_batch(subset)
                if self.cuda_available:
                    batch["input_dist"] = batch["input_dist"].cuda()
                    batch["target"] = batch["target"].cuda()
                # Prediction

                output = self.forward(batch)

                #get the value ordering of the output
                _, ordered_output_val = output.sort(dim=1, descending=True)
                #look up the value of the ordered index
                _, outputval_rank = ordered_output_val.sort(dim=1)

                target = []
                for val in batch["target"].cpu().data.numpy():
                   for n in val:
                       target.append(n)
                ans = [x for x, z in enumerate(target) if z ==1]
                #ans = batch["target"]
                sum_ans.append(output[0][ans].cpu().data.numpy().tolist()[0])
                success_flag = 0

                #print("{},is at {}th output with value {}".format(ans,outputval_rank[0][ans]+1,output[0][ans]))
                hit1= [] 
                mrr1 = []
                topk1 = []
               # print("output top1:", (torch.topk(output,1, sorted=True).indices).cpu().data.numpy().tolist())

                for i in (torch.topk(output,1, sorted=True).indices).cpu().data.numpy().tolist():
                    for j in i:
                        topk1.append(j)
                for a in ans:
                    if a in topk1:
                        hit1.append(1)
                        mrr1.append(1)
                    else:
                        hit1.append(0)
                        mrr1.append(0)
                #print("hit1", hit1)
                #print("mrr1", mrr1)
                
                avg_hit1.append(sum(hit1) / len(hit1))
                avg_mrr1.append(sum(mrr1) / len(mrr1))

                #print("avg_hit1", avg_hit1)
                #print("avg_mrr1", avg_mrr1)

                hit3= [] 
                mrr3 = []
                topk3 = []
                for i in (torch.topk(output,3, sorted=True).indices).cpu().data.numpy().tolist():
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
                
                #print("topk3",topk3)
                #print("hit",avg_hit3)
                #print("mrr",avg_mrr3)

                hit5= [] 
                mrr5 = []
                topk5 = []
                for i in (torch.topk(output,5, sorted=True).indices).cpu().data.numpy().tolist():
                    for j in i:
                        topk5.append(j)
                        #print("top{}: {}, with value {}".format(i.index(j)+1,j,output[0][j]))
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

                #print("topk5",topk5)
                #print("hit",avg_hit5)
                #print("mrr",avg_mrr5)

                hit10= [] 
                mrr10 = []
                topk10 = []
                for i in (torch.topk(output,10, sorted=True).indices).cpu().data.numpy().tolist():
                    for j in i:
                        topk10.append(j)
                        #print("top{}: {}, with value {}".format(i.index(j)+1,j,output[0][j]))
                for a in ans:
                    if a in topk10:
                        hit10.append(1)
                        mrr10.append(1/(1+topk10.index(a)))
                    else:
                        hit10.append(0)
                        mrr10.append(0)
                if(any(ele == 1 for ele in hit10)):
                    avg_hit10.append(1)
                else:
                    avg_hit10.append(0)
                avg_mrr10.append(max(mrr10))

                if(any(ele == 1 for ele in hit1)):
                    success_flag =1

                with open(os.path.join('./models/review_weighted_rec/final/seq/pretrain_fixed/', "pretrainfixed_testing_ht1"), "a+") as f:
                   text = "{}, {}, {}, {}, {}, {}\n".format(success_flag, counter, ans,outputval_rank[0][ans]+1,output[0][ans],batch["popularity"][0])
                   f.write(text)

                #with open(os.path.join('./models/review_weighted_rec/final/seq/initial/', "fail_initial_testing_ht1_popularityunder10"), "a+") as f:
                #    text = "{}, {}, {}, {}, {}\n".format(success_flag, counter, topk1,output[0][topk1],batch["popularity"][0])
                #    f.write(text)
                counter+=1

        print("HR @1 on test set is" , sum(avg_hit1) / len(avg_hit1))
        print("HR @3 on test set is" , sum(avg_hit3) / len(avg_hit3))
        print("HR @5 on test set is" , sum(avg_hit5) / len(avg_hit5))
        print("HR @10 on test set is" , sum(avg_hit10) / len(avg_hit10))

        print("MRR @1 on test set is" , sum(avg_mrr1) / len(avg_mrr1))
        print("MRR @3 on test set is" , sum(avg_mrr3) / len(avg_mrr3))
        print("MRR @5 on test set is" , sum(avg_mrr5) / len(avg_mrr5))
        print("MRR @10 on test set is" , sum(avg_mrr10) / len(avg_mrr10))
        print("average socre on the answer" , sum(sum_ans) / len(sum_ans))

class ReconstructionLoss(nn.Module):
    def __init__(self):
        super(ReconstructionLoss, self).__init__()
        if torch.cuda.is_available():
            self.cuda()

    def forward(self, model_output, target):
        #bce calculate each label's result, inclusing the movies marked as 0
        # loss = nn.BCEWithLogitsLoss(size_average=False, reduce=False)
        # output_loss = loss(model_output.float(), target.float())

        #nll loss learns the relations between the liked movies, consider loss only on the ans position
        #nll loss function contains only a -, therefore we need to add log by ourselves
        loss = nn.NLLLoss()

        #trandform from bce loss
        target_input = []
        for val in target.cpu().data.numpy():
            not0 = [x for x, z in enumerate(val) if z ==1]
            target_input.append(not0[0])

        ans = torch.LongTensor(target_input).cuda()
        output_loss = loss(torch.log(model_output),ans)

        return output_loss