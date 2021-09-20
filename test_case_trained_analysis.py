import pandas as pd
import re
import numpy as np
import os
import json
import csv 
import itertools
import pickle
from collections import Counter
import matplotlib.pyplot as plt

# this file is used for classifying the test generation converation
# by constructing the test file from those mentioned movies having
# no reviews/ 1-9 reviews/ 10 reviews up

#read the dataset
def load_data(path):
    #載入test資料
    """
    :param path:
    :return:
    """
    data = []
    for line in open(path,encoding="utf-8"):
        data.append(json.loads(line))
    return data

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

data_path = './redial/'
train_path = "train_data"
test_path="len1_TrainValidTest"
movie_path = './redial/movies_merged.csv'
train_data_db ={"train": os.path.join(data_path, train_path)}
conversation_train_data = {key: load_data(val) for key, val in train_data_db.items()}
test_data_db ={"test": os.path.join(data_path, test_path)}
conversation_test_data = {key: load_pickle_data(val) for key, val in test_data_db.items()}
id2name, db2id = get_movies(movie_path)
#global電影名稱對應id2name
db2name = {db: id2name[id] for db, id in db2id.items()}
n_movies = len(db2id.values())
#load database
#review_base = pd.read_csv('./external_knowledge_base/review_knowledge_base_IMDb.csv')
review_base = pd.read_csv('./external_knowledge_base/review_knowledge_base.csv')


def extract_trained_movie(data):
    form_data = []

    for (i, conversation) in enumerate(data):
        init_q = conversation["initiatorQuestions"]
            #存入seeker資料
        resp_q = conversation["respondentQuestions"]
            #存入recommender資料
        movies = conversation["movieMentions"]

        gen = (key for key in init_q if key in resp_q and key in movies and not db2name[int(key)].isspace())
            #取得和db有關電影

        # get mentioned movie list
        for key in gen:
            if((resp_q[key]["suggested"] == 1) | (init_q[key]["suggested"] == 1) | (resp_q[key]["suggested"] == 0) | (init_q[key]["suggested"] == 0)):
                form_data.append(db2id[int(key)])
    #counter
    print(Counter(form_data))
    plt.bar(list(Counter(form_data).keys()), Counter(form_data).values(), color='g')
    plt.show()
    
    #remove duplicate               
    #form_data = list(dict.fromkeys(form_data))


    return form_data

def extract_tested_movie(data):
    form_data = []

    for (i, conversation) in enumerate(data):
        init_q = conversation["initiatorQuestions"]
            #存入seeker資料
        resp_q = conversation["respondentQuestions"]
            #存入recommender資料
        movies = conversation["movieMentions"]

        gen = (key for key in init_q if key in resp_q and key in movies and not db2name[int(key)].isspace())
            #取得和db有關電影

        # get mentioned movie list
        for key in gen:
            if((resp_q[key]["suggested"] == 1) | (init_q[key]["suggested"] == 1)):
                form_data.append(db2id[int(key)])
                    
    form_data = list(dict.fromkeys(form_data))
    return form_data

if __name__ == '__main__':
    trained_movie_list = {key: extract_trained_movie(val) for key, val in conversation_train_data.items()}
    #trained_pool = []
    #for i in trained_movie_list.values():
    #    for id in i:
    #        trained_pool.append(id)
    
    #print(len(Counter(trained_pool).keys()))

    
    # counter = 0
    # tested_len = []
    # tested_movie_list = {key: extract_tested_movie(val) for key, val in conversation_test_data.items()}
    # for i in tested_movie_list.values():
    #     for id in i:
    #         tested_len.append(id)
    #         if(id in trained_pool):
    #             counter = counter+1
    # print("trained_pool:{}".format(len(trained_pool)))
    # print("counter:{}".format(counter))
    # print("tested_len:{}".format(len(tested_len)))

    #print("\ntrained percentage of {} is {}".format(test_path, (counter/len(tested_len))))
