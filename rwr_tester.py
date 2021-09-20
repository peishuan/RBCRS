import os
import config
from tqdm import tqdm
import torch
import csv
import re
from torch.autograd import Variable
import torch.nn.functional as F
import argparse
import pandas as pd
from models.review_weighted_recommender import Review_weighted_recommender
from utils import load_model
from beam_search import get_best_beam
import test_params
import pickle

def load_pickle_data(path):
    #載入train/test/valid資料
    """
    :param path:
    :return:
    """
    a_file = open(path, "rb")
    data = pickle.load(a_file)
    
    return data

def get_movies(path):
    # 載入電影資料

    id2name = {}
    db2id = {}
    
    with open(path, 'r',encoding="utf-8") as f:
    #開啟檔案
        reader = csv.reader(f)
        # 以csv reader讀取
        date_pattern = re.compile(r'\(\d{4}\)')
        # remove date from movie name 去除電影年分

        for row in reader:
            if row[0] != "index":
            #如果第一行非index
                id2name[int(row[0])] = date_pattern.sub('', row[1])
                # id2name [index] = 電影名稱
                db2id[int(row[2])] = int(row[0])
                # db2id[databaseId] = index; db2id is a dictionary mapping ReDial movie Ids to global movieId
    del db2id[-1]
    print("loaded {} movies from {}".format(len(id2name), path))
    return id2name, db2id

class TestingBatchLoader(object):
    def __init__(self, sources, batch_size, 
                 training_size=-1,
                 movie_rep_matrix_path='./review_enhanced_movie_rep/expanded_movie_rep_I_A_like.pkl',
                 rwr_data_path="./external_knowledge_base/review_weighted_rec_training/experi/seq_notseen_popularitylessthan10",
                 rwr_test="test",
                 movie_path=config.MOVIE_PATH,
                 shuffle_data=False,
                 process_at_instanciation=False):
        # sources paramater: string "dialogue/sentiment_analysis [movie_occurrences] [movieIds_in_target]"
        self.sources = sources
        self.batch_size = batch_size
        self.batch_index = {"train": 0, "valid": 0, "test": 0}
        self.training_size = training_size
        self.rwr_test_data_path =  {"test": os.path.join(rwr_data_path, rwr_test)}
        self.movie_rep_matrix_path = movie_rep_matrix_path
        self.movie_path = movie_path
        self.shuffle_data = shuffle_data
        # if true, call extract_dialogue when loading the data. (for training on several epochs)
        # Otherwise, call extract_dialogue at batch loading. (faster for testing the code)
        self.process_at_instanciation = process_at_instanciation
        #初始化DialogueBatchLoader資料

        self._get_dataset_characteristics()

    def _get_dataset_characteristics(self):
        # load movies. "db" refers to the movieId in the ReDial dataset, whereas "id" refers to the global movieId
        # (after matching the movies with movielens in match_movies.py).
        # So db2id is a dictionary mapping ReDial movie Ids to global movieIds
        self.id2name, self.db2id = get_movies(self.movie_path)
        #global電影名稱對應id2name
        self.db2name = {db: self.id2name[id] for db, id in self.db2id.items()}
        #ReDial電影名稱對應id2name
        self.n_movies = len(self.db2id.values())
        # number of movies mentioned in ReDial
        print('{} movies'.format(self.n_movies))
        self.movie_rep_matrix = load_pickle_data(self.movie_rep_matrix_path)
        # load data
        print("Loading and processing data")

        if "RW_rec" in self.sources:
            self.re_rec_data = {key: load_pickle_data(val) for key, val in self.rwr_test_data_path.items()}
            print("load re_rec_data from:{}".format(self.rwr_test_data_path))

        if "RW_rec" in self.sources:
            data = self.re_rec_data

        if self.shuffle_data:
            # shuffle each subset
            for _, val in data.items():
                shuffle(val)
        self.n_batches = {key: len(val) // self.batch_size for key, val in data.items()}

    def _load_rw_rec_batch(self, subset, flatten_messages=True, cut_dialogues=-1):
        batch = {"convId": [], "movieName": [], "input_dist": [],"target": [], "popularity":[]}

        # get batch
        batch_data = self.re_rec_data[subset][self.batch_index[subset] * self.batch_size:
                                               (self.batch_index[subset] + 1) * self.batch_size]
        for i, example in enumerate(batch_data):
            batch['convId'].append(example['convID'])
            batch['movieName'].append(example['likemoviename'])
            batch['popularity'].append(example['popularity'])
            for item in example["input_dist"]:
                batch["input_dist"].append(item)
            for item in example['target']:
                batch["target"].append(item)
            
        batch["target"] = Variable(torch.LongTensor(batch["target"]))  # (batch, 6)
        batch["input_dist"] = Variable(torch.DoubleTensor(batch["input_dist"]))  # (batch, 6)
        return batch

    def load_batch(self, subset="train",
                   flatten_messages=True, batch_input="random_noise", cut_dialogues=-1, max_num_inputs=None):
        """
        Get next batch
        :param batch_input:
        :param cut_dialogues:
        :param max_num_inputs:
        :param subset: "train", "valid" or "test"
        :param flatten_messages: if False, load the conversation messages as they are. If True, concatenate consecutive
        messages from the same sender and put a "\n" between consecutive messages.
        :return: batch
        """
        if "RW_rec" in self.sources:
            batch = self._load_rw_rec_batch(subset, flatten_messages, cut_dialogues=cut_dialogues)

        self.batch_index[subset] = (self.batch_index[subset] + 1) % self.n_batches[subset]

        return batch
    
if __name__ == '__main__':

    batch_loader = TestingBatchLoader(
                sources="RW_rec", # "sentiment_analysis movie_occurrences"
                batch_size=test_params.train_review_rep_params['batch_size'] # 1
            )
    #設定model
    rwr = Review_weighted_recommender(
        n_movies=batch_loader.n_movies,
        movie_rep_matrix=batch_loader.movie_rep_matrix, 
        params=test_params.review_rep_params
    )

    load_model(rwr,'./models/review_weighted_rec/final/seq/initial/checkpoint')
    rwr.test(batch_loader, subset="test")

