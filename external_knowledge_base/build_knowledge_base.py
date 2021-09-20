import pandas as pd
import numpy as np
import os
import glob
import json
import gzip
from itertools import islice
import collections
import re 
from tqdm import tqdm 

#this file read movie from IMDb, amazon to enrich the input information to autowncoder recommender

# ########################載入IMDb資料##############################################################
# #specify the path to IMDb knowledge base
# filepath = './IMDb_review_dataset/2_reviews_per_movie_raw/'
filepath = r'C:\2080_conversation_recommendation\external_knowledge_base\IMDb_review_dataset\2_reviews_per_movie_raw//'

# #acquire all file name
all_file = glob.glob(filepath+'*.csv')

# #read all file into list of dataframe
df_from_file = [pd.read_csv(f) for f in all_file]

# #add one column to the corresponding review that erase the "year.csv"
for dataframe, filename in zip(df_from_file,all_file):
        dataframe['movie_name'] = os.path.basename(filename)

# #get the concated dataframe
IMDb_df = pd.concat(df_from_file, ignore_index=True)

# #concat the title to the context as review
IMDb_df['review'] =IMDb_df['title'] + " " +IMDb_df['review'] 

# #drop unnecessaty columns
IMDb_df = IMDb_df.drop(['title','username','date','helpful','total'],axis=1)

# #dealing with missing data: 補1 
# # IMDb_df['rating'].loc[IMDb_df['rating'] == 'Null'] = 1
# #dealing with missing data: 補random
IMDb_df['rating'].loc[IMDb_df['rating'] == 'Null'] = np.random.randint(1,10)

# #rating數值轉數字
IMDb_df['rating'] = pd.to_numeric(IMDb_df['rating'])

# #以各電影一列
#IMDb_df = IMDb_df.groupby('movie_name',as_index=False).agg({'review':' '.join, 'rating':'mean'})


# #rease year.csv in title
IMDb_df["movie_name"] = IMDb_df["movie_name"].str.replace(".csv","")
IMDb_df["movie_name"] = IMDb_df["movie_name"].str.replace(' \d+', '')

IMDb_statistic = IMDb_df.groupby(['movie_name'],as_index = False)['review'].count()
print("review counts dataframe in IMDb dataset: ")
print(IMDb_statistic.head())
print("review counts statistic in IMDb dataset: ")
print(IMDb_statistic.describe())

#prepare data cleaning
review_list = IMDb_df["review"].tolist()
new_sen = []
replaced_review = []

#process data cleaning step: to lower, erase special symbol
for review in review_list:
    for word in review.split(" "):
        word = word.lower()
        match_pattern = re.findall(r'\b[a-z]{1,15}\b', word)
        new_sen.append(''.join(match_pattern))
    replaced_review.append(' '.join(new_sen))
    new_sen = []

#add back into dataframe
IMDb_df['replaced_review_text'] = replaced_review

#count length of review
IMDb_df["review_text_len"] = [len(x.split()) for x in IMDb_df['replaced_review_text'].tolist()]
IMDb_df = IMDb_df.drop(['review'], axis = 1)
print("review text counts statistic in IMDb dataset: ")
print(IMDb_df["review_text_len"].describe())

# # ##############################載入amazon資料##############################################################

# # #open Amazon meta ifle
# with open('./meta_Movies_and_TV.json','r') as f:
#         amazon_meta = pd.DataFrame(json.loads(line) for line in f)

# # # #amazon dfs delete unnecessary columns
# amazon_meta = amazon_meta.drop(['category', 'tech1', 'description', 'fit',  'also_buy', 'image',
#        'tech2', 'brand', 'feature', 'rank', 'also_view', 'main_cat',
#        'similar_item', 'date', 'price',  'details'],axis = 1)

# #open Amazon movie review ifle
# # Initial empty dataframe to store info
# num_file = sum([1 for i in open("./Movies_and_TV.json", "r")])
# amazon_df = pd.DataFrame(columns=['asin','review','rating'])
# with open('./Movies_and_TV.json','r') as f:
#     for line in tqdm(f, total=num_file):
#     # for line in f:
#         data = json.loads(line)
#         #取10篇
#         #if((amazon_df[amazon_df.asin==data['asin']].shape[0]) <10):
        
#         #全部存入
#         amazon_df = amazon_df.append({'asin':data['asin'], 'review':data.get('reviewText'),'rating':data['overall']},ignore_index=True)
#         #print(amazon_df[amazon_df.asin==data['asin']].shape[0]) 

# # #dealing with missing data: 補1 
# # # amazon_df['rating'].loc[amazon_df['rating'] == 'Null'] = 1
# # #dealing with missing data: 補random
# amazon_df['rating'].loc[amazon_df['rating'] == 'NaN'] = np.random.randint(1,5)

# # #以各電影一列
# #amazon_df = amazon_df.fillna('').groupby('asin',as_index=False).agg({'review':' '.join, 'rating':'mean'})

# # #合併電影id名稱及評論
# amazon_df = pd.merge(amazon_df,amazon_meta).rename(columns ={'rating':'rating','review':'review','title':'movie_name'})
# amazon_df = amazon_df.drop(['asin'],axis =1)

# # #改變欄順序跟IMDb合併
# cols = ['movie_name','review','rating']
# amazon_df = amazon_df[cols]

# amazon_df_statistic = amazon_df.groupby(['movie_name'],as_index = False)['review'].count()
# print("review counts dataframe in Amazon dataset: ")
# print(amazon_df_statistic.head())
# print("review counts statistic in Amazon dataset: ")
# print(amazon_df_statistic.describe())

# #prepare data cleaning
# review_list = amazon_df["review"].tolist()
# new_sen = []
# replaced_review = []

# #process data cleaning step: to lower, erase special symbol
# for review in review_list:
#     for word in str(review).split(" "):
#         word = word.lower()
#         match_pattern = re.findall(r'\b[a-z]{1,15}\b', word)
#         new_sen.append(''.join(match_pattern))
#     replaced_review.append(' '.join(new_sen))
#     new_sen = []

# #add back into dataframe
# amazon_df['replaced_review_text'] = replaced_review

# #count length of review
# amazon_df["review_text_len"] = [len(x.split()) for x in amazon_df['replaced_review_text'].tolist()]
# amazon_df = amazon_df.drop(['review'], axis = 1)
# print("review text counts statistic in Amazon dataset: ")
# print(amazon_df["review_text_len"].describe())


# # print(amazon_df.head())
# # print("Get {} movie from Amazon review dataset".format(amazon_df['movie_name'].count()))


# # ####################################合併兩者###########################################################################
# review_base_df = pd.concat([IMDb_df,amazon_df])
review_base_df = IMDb_df

# # #以各電影一列
# #review_base_df = review_base_df.groupby('movie_name',as_index=False).agg({'review':' '.join, 'rating':'mean'})

# review_base_df_statistic = review_base_df.groupby(['movie_name'],as_index = False)['rating'].count()
# print("review counts dataframe in merged dataset: ")
# print(review_base_df_statistic.head())
# print("review counts statistic in merged dataset: ")
# print(review_base_df_statistic.describe())

# #count length of review
# print("review text counts statistic in IMDb + Amazon dataset: ")
# print(review_base_df["review_text_len"].describe())

# #store result to file
review_base_df.to_csv('review_knowledge_base_IMDb.csv')

###################################查找DB對應 review id#################################################################

# # #讀入db檔
# movie_db = pd.read_csv('./redial/movies_merged.csv')

# # #取得所有電影名稱
# list_movie = movie_db['movieName'].apply(lambda x: x[:-7]).tolist()

# for name in list_movie:
#     #如果db中的電影有對應的review
#     if(name in review_base_df.movie_name.values):
#         #取得評論的index
#         a = review_base_df.loc[review_base_df['movie_name'] == name].index[0]
#         #將評論的index加入db表中
#         movie_db.loc[movie_db['movieName'].apply(lambda x: x[:-7]) == name,'reviewId']=str(a)

# #store result to file
# movie_db.to_csv('movie_merged_with_reviewId.csv')
# print("Match with {} movie-review in DB".format(movie_db['reviewId'].count()))