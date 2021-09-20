from tqdm import tqdm
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import argparse
import pandas as pd
from models.review_rep_trainer import ReviewRepTrainer
from batch_loaders.batch_loader import DialogueBatchLoader
from utils import load_model
from beam_search import get_best_beam
import test_params
import pickle

if __name__ == '__main__':

    #設定model
    sources = "review_rep"

    batch_loader = DialogueBatchLoader(
                sources=sources, # "sentiment_analysis movie_occurrences"
                batch_size=test_params.train_review_rep_params['batch_size'] # 1
            )

    #sa = model_class(train_vocab=batch_loader.train_vocabulary,params=test_params.review_rep_params,n_movies=batch_loader.n_movies)

    rrt = ReviewRepTrainer(
        #batch_loader.train_vocabulary,
        #batch_loader.n_movies,
        #params=test_params.review_rep_params
        train_vocab=batch_loader.train_vocabulary,
        params=test_params.review_rep_params,
        n_movies=batch_loader.n_movies
    )
    load_model(rrt,'./models/review_I_A_rep/unifying_sampling_like_review5_try/checkpoint')
    batch_loader.set_word2id(rrt.encoder.word2id)

    review_data = ['train','valid','test']
    #movie_rep = pd.DataFrame(columns=['movie_name', 'rep'])
    movie_rep_dict = {}
    result = {}
    for val in review_data:
        print("val:{}".format(val))
        result = rrt.build_rep_evaluate(batch_loader=batch_loader,subset=val, batch_input="full")
        #print("movie_rep:{}".format(result))
        movie_rep_dict.update(result)
        #movie_rep = movie_rep.append(result)
    
    #movie_rep = movie_rep.drop_duplicates(subset=['movie_name'], keep='last')
    #print("num of movies:{}".format(movie_rep['movie_name'].count()))
    #movie_rep.to_csv("movie_rep_matrix_I.csv")
    print(len(movie_rep_dict))
    try: 
        geeky_file = open('./review_enhanced_movie_rep/movie_rep_matrix_I_A_like_review5_try_dict.pkl', 'wb') 
        pickle.dump(movie_rep_dict, geeky_file) 
        geeky_file.close() 
  
    except: 
        print("Something went wrong")
        

        
    
