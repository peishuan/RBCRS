import pandas as pd
import re
import numpy as np
import os
import json
import csv 
import itertools
import pickle

# this file is used for preparing the training data for knowledge enhanced autoencoder
# first extract the liked movies in each conversation 
# then get the permutation of those movies by line
# in each permutation, get the first element and sample 10 corresponging reviews as input
# the output of training is the multi-hot representation of the liked movies over all movies.
# {reviews:{[r1], [r2], ... , [r10]}, target:{[0, 1 , 1 , 1 ,....,0]}}

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
data_path_db ={ "train_like": os.path.join(data_path, train_path),
            "valid_like": os.path.join(data_path, valid_path),
            "test_like": os.path.join(data_path, test_path)}
conversation_data = {key: load_data(val) for key, val in data_path_db.items()}
id2name, db2id = get_movies(movie_path)
#global電影名稱對應id2name
db2name = {db: id2name[id] for db, id in db2id.items()}
n_movies = len(db2id.values())
#load database
#review_base = pd.read_csv('./external_knowledge_base/review_knowledge_base_IMDb.csv')
review_base = pd.read_csv('./external_knowledge_base/review_knowledge_base.csv')

def extract_like_movie_review(data): 
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
             
            # get liked movie list
            likedmovielist = []
            for key in gen:
                if ((init_q[key]["liked"] == 1) & (resp_q[key]["liked"] == 1)):
                    likedmovielist.append(db2id[int(key)])
            
            #build target 
            target_list = [0] * n_movies
            for mid in likedmovielist:
                target_list[int(mid)] = 1 
            
            for mid in likedmovielist:
                target_sample = {"convID": [], "likemoviename":[], "target": []} 
                target_sample['convID'].append(convID)
                target_sample['likemoviename'].append(id2name[int(mid)].strip())
                target_sample['target'].append(target_list)
                form_data.append(target_sample)
        return form_data

def sample_reviews(data):
    # #unifying sampling

    form_data = []
    for idx, moviepair in enumerate(data):
        target_sample = {"convID": [], "likemoviename":[], "reviewCount":[], "reviews":[], "target": []} 
        tmp_df = review_base.loc[review_base['movie_name'] == moviepair['likemoviename'][0]]

        target_sample['convID'].append(moviepair['convID'][0])
        target_sample['likemoviename'].append(moviepair['likemoviename'][0])
        target_sample['target'].append(moviepair['target'][0])

        #there is corresponding review in knowledge base
        if(not tmp_df.empty):
            #過濾空評論
            review_df = tmp_df[tmp_df['review_text_len']>4]
            #review_df = tmp_df.dropna()
            #print(review_df)
            target_sample['reviewCount'].append(review_df['replaced_review_text'].count())
            # if((tmp_df['replaced_review_text'].count())/10 > 5 ): 決定sample次數
            if(review_df['replaced_review_text'].count()>=10):
                # 是否使用rating 作為挑選review依據
                sampleData = review_df.sample(n=10,replace=True, random_state=1)
                sampled_review_list = sampleData['replaced_review_text'].tolist()
                target_sample['reviews'].append(sampled_review_list)
            else:
                #under 10 review, pass
                continue
                # review_list = review_df['replaced_review_text'].tolist()
                # appendix = 10 - (review_df['replaced_review_text'].count())
                # for i in range(0,appendix):
                #     review_list.append("no_reviews")
                # #print(len(review_list))
                # target_sample['reviews'].append(review_list)

        #there is no review in knowledge base
        else:
            #no any review, pass
            continue
            # random initial movie representation
            # target_sample['reviewCount'].append(0)
            #target_sample['reviews'].append("NO_REVIEWS")
            #print("NO_REVIEWS")
        #print(target_sample)    
        form_data.append(target_sample)
    #-----------------------------------------------------------------

#    # # different amount sampling
#     form_data = []
#     for idx, moviepair in enumerate(data):
#         # target_sample = {"convID": [], "likemoviename":[], "reviewCount":[], "reviews":[], "target": []} 
#         tmp_df = review_base.loc[review_base['movie_name'] == moviepair['likemoviename'][0]]
#         # target_sample['convID'].append(moviepair['convID'][0])
#         # target_sample['likemoviename'].append(moviepair['likemoviename'][0])
#         # target_sample['target'].append(moviepair['target'][0])

#         #there is corresponding review in knowledge base
#         if(not tmp_df.empty):
#             #過濾空評論
#             review_df = tmp_df[tmp_df['review_text_len']>4]
#             if(review_df['replaced_review_text'].count()>=10):
#                 #評論數量小於平均值(50篇)
#                 if(review_df['replaced_review_text'].count()<30):
#                     # print("<60")
#                     for i in range(int((tmp_df['replaced_review_text'].count())/10)):

#                         target_sample = {"convID": [], "likemoviename":[], "reviewCount":[], "reviews":[], "target": []} 
#                         target_sample['convID'].append(moviepair['convID'][0])
#                         target_sample['likemoviename'].append(moviepair['likemoviename'][0])
#                         target_sample['target'].append(moviepair['target'][0])
#                         target_sample['reviewCount'].append(review_df['replaced_review_text'].count())

#                         sampleData = review_df.sample(n=10,replace=False, random_state=np.random.RandomState(0))
#                         sampled_review_list = sampleData['replaced_review_text'].tolist()
#                         target_sample['reviews'].append(sampled_review_list)
#                         form_data.append(target_sample)
#                         # print(len(target_sample['reviews'][0]))
#                 #評論數量大於30篇, sample 3次
#                 else:
#                     # print(">60")
#                     for i in range(3):
#                         target_sample = {"convID": [], "likemoviename":[], "reviewCount":[], "reviews":[], "target": []} 
#                         target_sample['convID'].append(moviepair['convID'][0])
#                         target_sample['likemoviename'].append(moviepair['likemoviename'][0])
#                         target_sample['target'].append(moviepair['target'][0])
#                         target_sample['reviewCount'].append(review_df['replaced_review_text'].count())

#                         sampleData = review_df.sample(n=10,replace=False, random_state=np.random.RandomState(0))
#                         sampled_review_list = sampleData['replaced_review_text'].tolist()
#                         target_sample['reviews'].append(sampled_review_list)
#                         form_data.append(target_sample)
#                         # print(len(target_sample['reviews'][0]))
#             else:
#                 # print("<10")
#                 review_list = review_df['replaced_review_text'].tolist()
#                 appendix = 10 - (review_df['replaced_review_text'].count())
#                 for i in range(0,appendix):
#                     review_list.append("no_reviews")
#                 #print(len(review_list))
#                 target_sample = {"convID": [], "likemoviename":[], "reviewCount":[], "reviews":[], "target": []} 
#                 target_sample['convID'].append(moviepair['convID'][0])
#                 target_sample['likemoviename'].append(moviepair['likemoviename'][0])
#                 target_sample['reviewCount'].append(review_df['replaced_review_text'].count())
#                 target_sample['target'].append(moviepair['target'][0])
#                 target_sample['reviews'].append(review_list)
#                 form_data.append(target_sample)
#                 # print(len(target_sample['reviews'][0]))
#         #there is no review in knowledge base
#         else:
#             # print("=0")
#             # random initial movie representation
#             # target_sample['reviewCount'].append(0)
#             target_sample = {"convID": [], "likemoviename":[], "reviewCount":[], "reviews":[], "target": []} 
#             target_sample['convID'].append(moviepair['convID'][0])
#             target_sample['likemoviename'].append(moviepair['likemoviename'][0])
#             target_sample['target'].append(moviepair['target'][0])
#             target_sample['reviews'].append("NO_REVIEWS")
#             form_data.append(target_sample)
#             #print(len(target_sample['reviews'][0]))
    #---------------------------------------------------------------------------
    # # #unifying sampling with high rating
    # form_data = []
    # for idx, moviepair in enumerate(data):
    #     target_sample = {"convID": [], "likemoviename":[], "reviewCount":[], "reviews":[], "target": []} 
    #     tmp_df = review_base.loc[review_base['movie_name'] == moviepair['likemoviename'][0]]

    #     target_sample['convID'].append(moviepair['convID'][0])
    #     target_sample['likemoviename'].append(moviepair['likemoviename'][0])
    #     target_sample['target'].append(moviepair['target'][0])

    #     #there is corresponding review in knowledge base
    #     if(not tmp_df.empty):
    #         #過濾空評論
    #         review_df = tmp_df[tmp_df['review_text_len']>4]
    #         #過濾出評分高的評論來抽樣
    #         review_df = tmp_df[tmp_df['rating']>=5]

    #         target_sample['reviewCount'].append(review_df['replaced_review_text'].count())
    #         # if((tmp_df['replaced_review_text'].count())/10 > 5 ): 決定sample次數
    #         if(review_df['replaced_review_text'].count()>=10):
    #             # 是否使用rating 作為挑選review依據
    #             sampleData = review_df.sample(n=10,replace=True, random_state=1)
    #             sampled_review_list = sampleData['replaced_review_text'].tolist()
    #             target_sample['reviews'].append(sampled_review_list)
    #         else:
    #             review_list = review_df['replaced_review_text'].tolist()
    #             appendix = 10 - (review_df['replaced_review_text'].count())
    #             for i in range(0,appendix):
    #                 review_list.append("no_reviews")
    #             #print(len(review_list))
    #             target_sample['reviews'].append(review_list)

    #     #there is no review in knowledge base
    #     else:
    #         # random initial movie representation
    #         # target_sample['reviewCount'].append(0)
    #         target_sample['reviews'].append("NO_REVIEWS")
    #         #print("NO_REVIEWS")
    #     #print(target_sample)    
    #     form_data.append(target_sample)
    #------------------------------------------------------------------
    # #unifying sampling with low rating
    # form_data = []
    # for idx, moviepair in enumerate(data):
    #     target_sample = {"convID": [], "likemoviename":[], "reviewCount":[], "reviews":[], "target": []} 
    #     tmp_df = review_base.loc[review_base['movie_name'] == moviepair['likemoviename'][0]]

    #     target_sample['convID'].append(moviepair['convID'][0])
    #     target_sample['likemoviename'].append(moviepair['likemoviename'][0])
    #     target_sample['target'].append(moviepair['target'][0])

    #     #there is corresponding review in knowledge base
    #     if(not tmp_df.empty):
    #         #過濾空評論
    #         review_df = tmp_df[tmp_df['review_text_len']>4]
    #         #過濾出評分高的評論來抽樣
    #         review_df = tmp_df[tmp_df['rating']<=3]

    #         target_sample['reviewCount'].append(review_df['replaced_review_text'].count())
    #         # if((tmp_df['replaced_review_text'].count())/10 > 5 ): 決定sample次數
    #         if(review_df['replaced_review_text'].count()>=10):
    #             # 是否使用rating 作為挑選review依據
    #             sampleData = review_df.sample(n=10,replace=True, random_state=1)
    #             sampled_review_list = sampleData['replaced_review_text'].tolist()
    #             target_sample['reviews'].append(sampled_review_list)
    #         else:
    #             review_list = review_df['replaced_review_text'].tolist()
    #             appendix = 10 - (review_df['replaced_review_text'].count())
    #             for i in range(0,appendix):
    #                 review_list.append("no_reviews")
    #             #print(len(review_list))
    #             target_sample['reviews'].append(review_list)

    #     #there is no review in knowledge base
    #     else:
    #         # random initial movie representation
    #         # target_sample['reviewCount'].append(0)
    #         target_sample['reviews'].append("NO_REVIEWS")
    #         #print("NO_REVIEWS")
    #     #print(target_sample)    
    #     form_data.append(target_sample)

    return form_data

def write_to_file(name, val):
    # finished = []
    # write_db ={ "train": "seen_and_likedmovie_review_data_train.pkl",
    #         "valid": "seen_and_likedmovie_review_data_valid.pkl",
    #         "test": "seen_and_likedmovie_review_data_test.pkl"}
    print(name)
    # print(val)
    a_file = open(os.path.join('./external_knowledge_base/like_movie_review/unifying_sampling/',name), "wb")
    pickle.dump(val, a_file)
    a_file.close()
    # finished = {key: }

    

#train:{{'convID': ['8108'], 'likemovie': [The Avengers], 'target': [[]], {}...,{}}, valid:{[],[]...[]},test:{[],[]...[]}
#挑出看過且喜歡的對話電影
seen_and_likedmovie_data = {key: extract_like_movie_review(val) for key, val in conversation_data.items()}
#為看過且喜歡的電影sample reviews
seen_and_likedmovie_review_data = {key: sample_reviews(val) for key, val in seen_and_likedmovie_data.items()}
#將看過且喜歡的電影資料training/valid/testing寫檔
write_to_file = {key: write_to_file(key, val) for key, val in seen_and_likedmovie_review_data.items()}
