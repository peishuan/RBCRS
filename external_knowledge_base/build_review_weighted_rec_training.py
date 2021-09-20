import pandas as pd
import re
import torch.nn as nn
import torch
import numpy as np
import os
import json
import csv 
import itertools
import pickle
from itertools import combinations
from collections import Counter
import nltk

# this file is used for preparing the training data for review weighted recommender
# first extract the liked movies in each conversation 
# the output of training is the multi-hot representation of the liked movies over all movies.
def load_pickle_data(path):
    #載入train/test/valid資料
    """
    :param path:
    :return:
    """
    a_file = open(path, "rb")
    data = pickle.load(a_file)
    
    return data

#read the dataset
def load_data(path):
    #載入train/test/valid資料
    """
    :param path:
    :return:
    """
    data = []
    for line in open(path,encoding="utf-8"):
        data.append(json.loads(line))
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
valid_path = "valid_data"
test_path = "test_data"
movie_path = './redial/movies_merged.csv'
data_path_db ={ "train": os.path.join(data_path, train_path),
            "valid": os.path.join(data_path, valid_path),
            "test": os.path.join(data_path, test_path)}
train_data_db ={"train": os.path.join(data_path, train_path)}
conversation_data = {key: load_data(val) for key, val in data_path_db.items()}
conversation_train_data = {key: load_data(val) for key, val in train_data_db.items()}
id2name, db2id = get_movies(movie_path)
#global電影名稱對應id2name
db2name = {db: id2name[id] for db, id in db2id.items()}
n_movies = len(db2id.values())
#load database
#review_base = pd.read_csv('./review_knowledge_base_IMDb.csv')
#review_base = pd.read_csv('./external_knowledge_base/review_knowledge_base.csv')

def tokenize(message):
    #句子斷字 轉小寫
    """
    Text processing: Sentence tokenize, then concatenate the word_tokenize of each sentence. Then lower.
    :param message:
    :return:
    """
    sentences = nltk.sent_tokenize(str(message))
    tokenized = []
    for sentence in sentences:
        tokenized += nltk.word_tokenize(str(sentence))
    return [word.lower() for word in tokenized]

def extract_seen_like_movie(data): 
        form_data = []
        for (i, conversation) in enumerate(data):
            convID = conversation["conversationId"]
            #存入對話id
            init_q = conversation["initiatorQuestions"]
            #存入seeker資料
            resp_q = conversation["respondentQuestions"]
            #存入recommender資料
            gen = (key for key in init_q if key in resp_q and not db2name[int(key)].isspace())
            #取得和db有關電影
             
            # get seen and liked movie list
            likedmovielist = []
            for key in gen:
                if ((init_q[key]["seen"] == 1) & (init_q[key]["liked"] == 1) & (resp_q[key]["seen"] == 1) & (resp_q[key]["liked"] == 1)):
                    likedmovielist.append(db2id[int(key)])
            
            # #建立一對一的training 在每一個對話中提到的電影裡 看過且喜歡的一個要和另一個看過且喜歡的配對 訓練mlp可以預測一個喜歡的輸入時輸出另一個喜歡的
            # #確認建立一對一對應後還有可以推薦的項目
            # if (len(likedmovielist)>1):
            #     for mid in likedmovielist:
            #         tar_list = []
            #         #輸入不與輸出一樣
            #         for tar in likedmovielist:
            #             if tar != mid:
            #                 tar_list.append(tar)
                       
            #         #針對每一個一對一的輸出建立training pair
            #         for i in tar_list:
            #             target_sample = {"convID": [], "likemoviename":[],"input_dist":[], "target": []} 
            #             target_sample['convID'].append(convID)
            #             target_sample['likemoviename'].append(id2name[int(mid)].strip())
            #             #mid 一個輸入
            #             input_list = [0] * n_movies
            #             input_list[int(mid)] = 1  
            #             target_sample['input_dist'].append(input_list)
                            
            #             #對應一個輸出
            #             target_list = [0] * n_movies
            #             target_list[int(i)] = 1
            #             target_sample['target'].append(target_list)
            #             #美對伊一個一對一就送出一筆
            #             form_data.append(target_sample)
            # else:
            #     # print(likedmovielist)
            #     continue
            #-------------------------------------------------------------------------------------------------------------------------------
            #建立二對一的training 在每一個對話中提到的電影裡 看過且喜歡的兩個要和另一個看過且喜歡的配對 訓練mlp可以預測兩個喜歡的輸入時輸出另一個喜歡的
            #確認建立二對一對應後還有可以推薦的項目
            if (len(likedmovielist)>2):
                comb = list(combinations(likedmovielist,2))
                for mid in comb:
                    tar_list = []
                    #輸入不與輸出一樣
                    for tar in likedmovielist:
                        if (tar != mid[0]) & (tar != mid[1]):
                            tar_list.append(tar)

                    #針對每一個二對一的輸出建立training pair
                    for i in tar_list:
                        target_sample = {"convID": [], "likemoviename":[],"input_dist":[], "target": []} 
                        target_sample['convID'].append(convID)
                        target_sample['likemoviename'].append(id2name[int(mid[0])].strip())
                        target_sample['likemoviename'].append(id2name[int(mid[1])].strip())
                        #mid 兩個輸入
                        input_list = [0] * n_movies
                        input_list[int(mid[0])] = 1  
                        input_list[int(mid[1])] = 1  
                        target_sample['input_dist'].append(input_list)
                            
                        #對應一個輸出
                        target_list = [0] * n_movies
                        target_list[int(i)] = 1
                        target_sample['target'].append(target_list)
                        #美對伊一個一對一就送出一筆
                        form_data.append(target_sample)
            else:
                # print(likedmovielist)
                continue
        return form_data

def extract_likedmovie_in_sequence(data): 
        form_data = []
        for (i, conversation) in enumerate(data):
            convID = conversation["conversationId"] 
            init_q = conversation["initiatorQuestions"] #seeker answer on seeker preference
            resp_q = conversation["respondentQuestions"] #recommender answer on seeker preference

            #extract movie mention in seq
            movie_mention_seq = []
            seeker_like_seq =[]
            for message in conversation["messages"]:
            # remove "@" and add spaces around movie mentions to be sure to count them as single tokens
            # tokens that match /^\d{5,6}$/ are movie mentions
                pattern = re.compile(r'@(\d+)')
                message_text = pattern.sub(lambda m: " " + m.group(1) + " ", message["text"])
                text = tokenize(message_text)
                #extract movie ids
                index = 0
                pattern2 = re.compile(r'^\d{5,6}$')
                while index < len(text):
                    word = text[index]
                    # Check if word corresponds to a movieId.
                    if pattern2.match(word) and int(word) in db2id and not db2name[int(word)].isspace():
                        # get the global Id
                        if db2id[int(word)] not in movie_mention_seq:
                            movie_mention_seq.append(db2id[int(word)])
                        #check question exists
                        if ((isinstance(init_q,dict))&(isinstance(resp_q,dict))):
                            # check movie id in question
                            if((str(word) in init_q.keys()) & (str(word) in resp_q.keys())):
                                #check both seeker and recommender said seeker like
                                if((init_q[str(word)]['liked'] == 1) & (resp_q[str(word)]['liked'] == 1)):
                                    if db2id[int(word)] not in seeker_like_seq:
                                        seeker_like_seq.append(db2id[int(word)])
                        else:
                            print(word,"not in question/question not exists")
                    index+=1
                    
            # #建立一對一的training 在每一個對話中提到的電影裡 喜歡的一個要和另一個喜歡的配對 訓練mlp可以預測一個喜歡的輸入時輸出另一個喜歡的
            # #確認建立一對一對應後還有可以推薦的項目
            # if (len(seeker_like_seq)>1):
            #     # creat 1to1 combination for like seq
            #     like_combinations = list(itertools.combinations(seeker_like_seq, 2))
            #     #針對每一個一對一的輸出建立training pair
            #     for i in like_combinations:
            #         target_sample = {"convID": [], "likemoviename":[],"input_dist":[], "target": []} 
            #         target_sample['convID'].append(convID)
            #         target_sample['likemoviename'].append(id2name[int(i[0])].strip())
            #         #mid 一個輸入
            #         input_list = [0] * n_movies
            #         input_list[int(i[0])] = 1  
            #         target_sample['input_dist'].append(input_list)
                            
            #         #對應一個輸出
            #         target_list = [0] * n_movies
            #         target_list[int(i[1])] = 1
            #         target_sample['target'].append(target_list)
            #         #美對伊一個一對一就送出一筆
            #         form_data.append(target_sample)
            # else:
            #     # print(likedmovielist)
            #     continue
            #-------------------------------------------------------------------------------------------------------------------------------
            #建立二對一的training 在每一個對話中提到的電影裡 喜歡的兩個要和另一個喜歡的配對 訓練mlp可以預測兩個喜歡的輸入時輸出另一個喜歡的
            #確認建立二對一對應後還有可以推薦的項目
            if (len(seeker_like_seq)>2):
                like_combinations = list(itertools.combinations(seeker_like_seq, 2))
                for mid in like_combinations:
                    tar_list = []
                    #輸入不與輸出一樣and in order
                    for tar in seeker_like_seq:
                        if (tar != mid[0]) & (tar != mid[1])&(seeker_like_seq.index(tar)>seeker_like_seq.index(mid[0]))&(seeker_like_seq.index(tar)>seeker_like_seq.index(mid[1])):
                            tar_list.append(tar)

                    #針對每一個二對一的輸出建立training pair
                    for i in tar_list:
                        target_sample = {"convID": [], "likemoviename":[],"input_dist":[], "target": []} 
                        target_sample['convID'].append(convID)
                        target_sample['likemoviename'].append(id2name[int(mid[0])].strip())
                        target_sample['likemoviename'].append(id2name[int(mid[1])].strip())
                        #mid 兩個輸入
                        input_list = [0] * n_movies
                        input_list[int(mid[0])] = 1  
                        input_list[int(mid[1])] = 1  
                        target_sample['input_dist'].append(input_list)
                            
                        #對應一個輸出
                        target_list = [0] * n_movies
                        target_list[int(i)] = 1
                        target_sample['target'].append(target_list)
                        #美對伊一個一對一就送出一筆
                        form_data.append(target_sample)
            else:
                # print(likedmovielist)
                continue
        return form_data

def extract_likedmovie(data):

        trained_movie_list = {key: extract_trained_movie(val) for key, val in conversation_train_data.items()}  
        trained_pool = []
        for i in trained_movie_list.values():
            for id in i:
                trained_pool.append(id)
        trained_stastic = Counter(trained_pool) 

        form_data = []
        for (i, conversation) in enumerate(data):
            convID = conversation["conversationId"] 
            init_q = conversation["initiatorQuestions"] #seeker answer on seeker preference
            resp_q = conversation["respondentQuestions"] #recommender answer on seeker preference

            #extract movie mention in seq
            rec_mention_seeker_like = []
            seeker_like_seq =[]
            for message in conversation["messages"]:
            # remove "@" and add spaces around movie mentions to be sure to count them as single tokens
            # tokens that match /^\d{5,6}$/ are movie mentions
                pattern = re.compile(r'@(\d+)')
                message_text = pattern.sub(lambda m: " " + m.group(1) + " ", message["text"])
                text = tokenize(message_text)
                #extract movie ids
                index = 0
                pattern2 = re.compile(r'^\d{5,6}$')
                while index < len(text):
                    word = text[index]
                    # Check if word corresponds to a movieId.
                    if pattern2.match(word) and int(word) in db2id and not db2name[int(word)].isspace():
                        # get the global Id
                        #if db2id[int(word)] not in movie_mention_seq:
                        #    movie_mention_seq.append(db2id[int(word)])
                        #check question exists
                        if ((isinstance(init_q,dict))&(isinstance(resp_q,dict))):
                            # check movie id in question
                            if((str(word) in init_q.keys()) & (str(word) in resp_q.keys())):
                                #check both seeker and recommender said seeker like
                                if((init_q[str(word)]['liked'] == 1) & (resp_q[str(word)]['liked'] == 1)):
                                    if db2id[int(word)] not in seeker_like_seq:
                                        seeker_like_seq.append(db2id[int(word)])
                                #rec mentioned seeker liked        
                                if((init_q[str(word)]['suggested'] == 1) & (resp_q[str(word)]['suggested'] == 1) & (init_q[str(word)]['liked'] == 1) & (resp_q[str(word)]['liked'] == 1) & (init_q[str(word)]['seen'] == 0) & (resp_q[str(word)]['seen'] == 0)):
                                    if db2id[int(word)] not in rec_mention_seeker_like:
                                        rec_mention_seeker_like.append(db2id[int(word)])
                        else:
                            print(word,"not in question/question not exists")
                    index+=1

            # example: all_like[1,2,3,4,5,6]
            #          r_like[2,3,6]
            #    train: ([1,3,4,5,6][2])
            #           ([1,2,4,5,6][3])
            #           ([1,2,3,4,5][6]) 
            # #輸出: recommender 提到且seeker喜歡(R_like)中的一部電影
            # #輸入:  All_like - 輸出

            # # #no_seq
            # if (len(rec_mention_seeker_like)):
            #     for mid in rec_mention_seeker_like:
            #         #select testing case based on popularity
            #         if(trained_stastic[int(mid)] <=10):
            #             #針對每一個rec_men_seeker_like建立training pair
            #             target_sample = {"convID": [], "likemoviename":[],"input_dist":[], "target": [], "popularity":[]} 
            #             target_sample['convID'].append(convID)
            #             target_sample['likemoviename'].append(id2name[int(mid)].strip())
            #             #ALL_like-R_like
            #             all_like=[]
            #             input_list = [0] * n_movies
            #             for a in seeker_like_seq:
            #                 if(a!=mid):
            #                     all_like.append(a)
            #                     input_list[int(a)] = 1  
            #             target_sample['input_dist'].append(input_list)
                            
            #             #對應一個輸出
            #             target_list = [0] * n_movies
            #             target_list[int(mid)] = 1
            #             target_sample['target'].append(target_list)

            #             target_sample['popularity'].append(trained_stastic[int(mid)])
                        
            #             #美對伊一個一對一就送出一筆
            #             form_data.append(target_sample)
            #         else:
            #             continue

            #seq
            if (len(rec_mention_seeker_like)):
                for mid in rec_mention_seeker_like:
                #針對每一個rec_men_seeker_like建立training pair
                    if(trained_stastic[int(mid)] <=10):
                        target_sample = {"convID": [], "likemoviename":[],"input_dist":[], "target": [], "popularity":[]} 
                        target_sample['convID'].append(convID)
                        target_sample['likemoviename'].append(id2name[int(mid)].strip())
                        #ALL_like-R_like
                        all_like=[]
                        input_list = [0] * n_movies
                        for a in seeker_like_seq:
                            if((a!=mid) & (seeker_like_seq.index(a)<seeker_like_seq.index(mid))):
                                all_like.append(a)
                                input_list[int(a)] = 1  
                        target_sample['input_dist'].append(input_list)
                    
                        #對應一個輸出
                        target_list = [0] * n_movies
                        target_list[int(mid)] = 1
                        target_sample['target'].append(target_list)

                        target_sample['popularity'].append(trained_stastic[int(mid)])

                        #美對伊一個一對一就送出一筆
                        if(all_like):
                            form_data.append(target_sample)
                        else:
                            continue
                    else:
                        continue
                    
            else:
                continue
        return form_data

def monitor_rec_case(data):
    #input would be the mention movies in order:from the first to the one recommender mention seeker like
    #the output is to predict the answer movie id 
        form_data = []
        for (i, conversation) in enumerate(data):
            convID = conversation["conversationId"] 
            init_q = conversation["initiatorQuestions"] #seeker answer on seeker preference
            resp_q = conversation["respondentQuestions"] #recommender answer on seeker preference

            #extract movie mention in seq, from the first mention to the one rec mention ans seeker like
            #movie_mention_seq = []
            seeker_like_seq =[]
            ans_flag=0
            for message in conversation["messages"]:
            # remove "@" and add spaces around movie mentions to be sure to count them as single tokens
            # tokens that match /^\d{5,6}$/ are movie mentions
                pattern = re.compile(r'@(\d+)')
                message_text = pattern.sub(lambda m: " " + m.group(1) + " ", message["text"])
                text = tokenize(message_text)
                #extract movie ids
                index = 0
                pattern2 = re.compile(r'^\d{5,6}$')
                while index < len(text):
                    if(ans_flag):
                        break
                    else:
                        word = text[index]
                        # Check if word corresponds to a movieId.
                        if pattern2.match(word) and int(word) in db2id and not db2name[int(word)].isspace():
                            # get the global Id
                            # if db2id[int(word)] not in movie_mention_seq:
                            #     movie_mention_seq.append(db2id[int(word)])
                            #check question exists
                            if ((isinstance(init_q,dict))&(isinstance(resp_q,dict))):
                            # check movie id in question
                                if((str(word) in init_q.keys()) & (str(word) in resp_q.keys())):
                                    #ordinary mentioned movie
                                    if db2id[int(word)] not in seeker_like_seq:
                                        seeker_like_seq.append(db2id[int(word)])

                                    #check ans: recommender mention & seeker like
                                    if((init_q[str(word)]['suggested'] == 0) & (resp_q[str(word)]['suggested'] == 0) & (init_q[str(word)]['liked'] == 1) & (resp_q[str(word)]['liked'] == 1)):
                                        #if db2id[int(word)] not in seeker_like_seq:
                                        #seeker_like_seq.append(db2id[int(word)])
                                        print("find ans! break")
                                        print(db2id[int(word)])
                                        ans_flag=1
                            else:
                                print(word,"not in question/question not exists")
                    
                    index+=1
            # build the input output pair
            if (len(seeker_like_seq)>1):
                target_sample = {"convID": [], "likemoviename":[],"input_dist":[], "target": []} 
                target_sample['convID'].append(convID)
                #target_sample['likemoviename'].append(id2name[int(i[0])].strip())

                #build input
                input_list = [0] * n_movies
                for i in seeker_like_seq[0:len(seeker_like_seq)-1]:
                    input_list[int(i)] = 1  
                target_sample['input_dist'].append(input_list)
                            
                #對應一個輸出
                target_list = [0] * n_movies
                target_list[int(seeker_like_seq[len(seeker_like_seq)-1])] = 1
                target_sample['target'].append(target_list)
                #美對伊一個一對一就送出一筆
                form_data.append(target_sample)
            else:
                continue
        return form_data

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
             #check question exists
            if ((isinstance(init_q,dict))&(isinstance(resp_q,dict))):
            # check movie id in question
                if((str(key) in init_q.keys()) & (str(key) in resp_q.keys())):
                #check both seeker and recommender said seeker like
                    form_data.append(db2id[int(key)])
    return form_data


def write_to_file(name, val):
    print(name)
    a_file = open(os.path.join('./external_knowledge_base/review_weighted_rec_training/experi/seq_notseen_popularitylessthan10/',name), "wb")
    pickle.dump(val, a_file)
    a_file.close()

    

#train:{{'convID': ['8108'], 'likemovie': [The Avengers], 'target': [[]], {}...,{}}, valid:{[],[]...[]},test:{[],[]...[]}
#挑出看過且喜歡的對話電影
seen_and_likedmovie_data = {key: extract_likedmovie(val) for key, val in conversation_data.items()}

#挑出喜歡的對話電影in time sequential
#likedmovie_in_sequence = {key: extract_likedmovie_in_sequence(val) for key, val in conversation_data.items()}

#seeker like movie 
#likedmovie = {key:monitor_rec_case(val) for key, val in conversation_data.items()}

#為看過且喜歡的電影sample reviews
#seen_and_likedmovie_review_data = {key: sample_reviews(val) for key, val in seen_and_likedmovie_data.items()}
#將看過且喜歡的電影資料training/valid/testing寫檔
write_to_file = {key: write_to_file(key, val) for key, val in seen_and_likedmovie_data.items()}
