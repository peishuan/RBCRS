import pandas as pd
import numpy as np
import torch
from torch import nn
import pickle
# This file construct a embedding matrix for the predictor 
# it expand the movie representation from 4258(mentioned)*512 to 59946(all)*512
# we pad same random number within pretrain/rand matrix

# # #讀入db檔
movie_db = pd.read_csv('./redial/movies_merged.csv')

# #取得所有電影名稱
list_movie_o = movie_db['movieName'][0:6924].tolist()
movie_list =[]
for name in list_movie_o:
    if name[-1] != ')':
        movie_list.append(name)
    else:
        movie_list.append(name[:-7])

expanded_movie_rep_pretrain = []
a_file = open('./review_enhanced_movie_rep/movie_rep_matrix_I_A_like_review5_try_dict.pkl', "rb")
movie_rep_pretrain = pickle.load(a_file)

print(len(movie_rep_pretrain))
# expanded_movie_rep_rand = []
# b_file = open('./review_enhanced_movie_rep/movie_rep_matrix_I_A_review_only_randominit_dict.pkl', "rb")
# movie_rep_rand = pickle.load(b_file)

# expanded_movie_rep_totallyrand = []
#construct a 
w=torch.Tensor(1,512)
for name in movie_list:
    if(name in movie_rep_pretrain.keys()):
        expanded_movie_rep_pretrain.append(np.array(movie_rep_pretrain[name], dtype="float64"))
        #print(np.array(movie_rep_pretrain[name], dtype="float64").shape)
        #expanded_movie_rep_rand.append(np.array(movie_rep_rand[name], dtype="float64"))
        #random a matrix
        #expanded_movie_rep_totallyrand.append(np.array([np.random.rand(512,)], dtype="float64"))
        #expanded_movie_rep_totallyrand.append(np.array([np.zeros(512,)], dtype="float64"))
    else:
        #tmp_rand = np.array([np.random.rand(512,)], dtype="float64")

        #random a xvaier uniform dist
        tmp_rand = nn.init.xavier_uniform_(w,gain=1)
        expanded_movie_rep_pretrain.append(tmp_rand.numpy())
        #print(tmp_rand.numpy().shape)

        #expanded_movie_rep_rand.append(tmp_rand)
        #expanded_movie_rep_totallyrand.append(np.array([np.random.rand(512,)], dtype="float64"))
        #expanded_movie_rep_totallyrand.append(np.array([np.zeros(512,)], dtype="float64"))

expanded_movie_rep_pretrain = np.array(expanded_movie_rep_pretrain, dtype="float64")
#expanded_movie_rep_rand = np.array(expanded_movie_rep_rand, dtype="float64")
#expanded_movie_rep_totallyrand = np.array(expanded_movie_rep_totallyrand, dtype="float64")

print(len(expanded_movie_rep_pretrain))
#print(expanded_movie_rep_rand.shape)
#print(expanded_movie_rep_totallyrand)

try: 
    geeky_file = open('./review_enhanced_movie_rep/expanded_movie_rep_I_A_like_review5_try.pkl', 'wb') 
    pickle.dump(expanded_movie_rep_pretrain, geeky_file) 
    geeky_file.close() 

    # geeky_file2 = open('./review_enhanced_movie_rep/expanded_movie_rep_I_A_review_only_randinit_samerand.pkl', 'wb') 
    # pickle.dump(expanded_movie_rep_rand, geeky_file2) 
    # geeky_file2.close()

   # geeky_file3 = open('./review_enhanced_movie_rep/expanded_movie_rep_I_A_review_only_totallyrand_samerandzero.pkl', 'wb') 
   # pickle.dump(expanded_movie_rep_totallyrand, geeky_file3) 
   # geeky_file3.close() 
except: 
    print("Something went wrong")

# # movie_rep_I_A
# expanded_movie_rep_I_A = []
# a_file = open('./review_enhanced_movie_rep/movie_rep_matrix_I_A_different_dict.pkl', "rb")
# movie_rep_I_A = pickle.load(a_file)

# for name in movie_list:
#     if(name in movie_rep_I_A.keys()):
#         expanded_movie_rep_I_A.append(np.array(movie_rep_I_A[name], dtype="float64"))
#     else:
#         expanded_movie_rep_I_A.append(np.array([np.random.rand(512,)], dtype="float64"))

# expanded_movie_rep_I_A = np.array(expanded_movie_rep_I_A, dtype="float64")

# print(expanded_movie_rep_I_A.shape)
# try: 
#     geeky_file = open('./review_enhanced_movie_rep/expanded_movie_rep_I_A_different.pkl', 'wb') 
#     pickle.dump(expanded_movie_rep_I_A, geeky_file) 
#     geeky_file.close() 
# except: 
#     print("Something went wrong")
