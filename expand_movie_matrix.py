import pandas as pd
import numpy as np
import torch
import pickle
# This file construct a embedding matrix for the predictor 
# it expand the movie representation from 3774(mentioned)*512 to 59946(all)*512


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

expanded_movie_rep_I = []
a_file = open('./movie_rep_matrix_I_dict.pkl', "rb")
movie_rep_I = pickle.load(a_file)

for name in movie_list:
    if(name in movie_rep_I.keys()):
        expanded_movie_rep_I.append(np.array(movie_rep_I[name], dtype="float64"))
    else:
        expanded_movie_rep_I.append(np.array([np.random.rand(512,)], dtype="float64"))

expanded_movie_rep_I = np.array(expanded_movie_rep_I, dtype="float64")

print(expanded_movie_rep_I.shape)
try: 
    geeky_file = open('expand_movie_rep_I.pkl', 'wb') 
    pickle.dump(expanded_movie_rep_I, geeky_file) 
    geeky_file.close() 
except: 
    print("Something went wrong")

# # movie_rep_I_A
expanded_movie_rep_I_A = []
a_file = open('./movie_rep_matrix_I_A_dict.pkl', "rb")
movie_rep_I_A = pickle.load(a_file)

for name in movie_list:
    if(name in movie_rep_I.keys()):
        expanded_movie_rep_I_A.append(np.array(movie_rep_I_A[name], dtype="float64"))
    else:
        expanded_movie_rep_I_A.append(np.array([np.random.rand(512,)], dtype="float64"))

expanded_movie_rep_I_A = np.array(expanded_movie_rep_I_A, dtype="float64")

print(expanded_movie_rep_I_A.shape)
try: 
    geeky_file = open('expanded_movie_rep_I_A.pkl', 'wb') 
    pickle.dump(expanded_movie_rep_I_A, geeky_file) 
    geeky_file.close() 
except: 
    print("Something went wrong")
