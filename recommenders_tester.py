import os
import config
from tqdm import tqdm
import numpy as np
import torch
import csv
import re
import json
from torch.autograd import Variable
import torch.nn.functional as F
import argparse
import pandas as pd
from models.review_weighted_recommender import Review_weighted_recommender
from models.recommender_model import Recommender as Ori_Recommender
from models.enhanced_recommender_model import Recommender as Enhanced_Recommender
from utils import load_model
from beam_search import get_best_beam
import test_params
import pickle
from utils import tokenize, permute, tokenize_review

def load_pickle_data(path):
    #載入train/test/valid資料
    """
    :param path:
    :return:
    """
    a_file = open(path, "rb")
    data = pickle.load(a_file)
    
    return data

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

class TestingBatchLoader(object):
    def __init__(self, sources, batch_size, 
                 training_size=-1,
                 conversation_length_limit=config.CONVERSATION_LENGTH_LIMIT, # 每段對話小於40句
                 utterance_length_limit=config.UTTERANCE_LENGTH_LIMIT, # 每句不多於80字
                 movie_rep_matrix_path='./review_enhanced_movie_rep/expanded_movie_rep_I_A_like.pkl',
                 data_path=config.REDIAL_DATA_PATH,
                 test_path="test_data",#_ans_train_above0under10",
                 movie_path=config.MOVIE_PATH,
                 vocab_path=config.VOCAB_PATH,
                 shuffle_data=False,
                 process_at_instanciation=False):
        # sources paramater: string "dialogue/sentiment_analysis [movie_occurrences] [movieIds_in_target]"
        self.sources = sources
        self.batch_size = batch_size
        self.conversation_length_limit = conversation_length_limit
        self.utterance_length_limit = utterance_length_limit
        self.batch_index = {"train": 0, "valid": 0, "test": 0}
        self.training_size = training_size
        self.data_path = {"test": os.path.join(data_path, test_path)}
        self.movie_rep_matrix_path = movie_rep_matrix_path
        self.movie_path = movie_path
        self.vocab_path = vocab_path
        self.word2id = None
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
        print(self.movie_rep_matrix_path)
        self.movie_rep_matrix = load_pickle_data(self.movie_rep_matrix_path)
        # load data
        print("Loading and processing data")
        self.conversation_data = {key: load_data(val) for key, val in self.data_path.items()}
        
        # load vocabulary
        self.train_vocabulary = self._get_vocabulary()
        print("Vocabulary size : {} words.".format(len(self.train_vocabulary)))

        if "dialogue" in self.sources:
            data = self.conversation_data
            
        if self.shuffle_data:
            # shuffle each subset
            for _, val in data.items():
                shuffle(val)
        self.n_batches = {key: len(val) // self.batch_size for key, val in data.items()}

    def _get_vocabulary(self):
        #取得train data中單字
        """
        get the vocabulary from the train data
        :return: vocabulary
        """
        if os.path.isfile(self.vocab_path):
            #如果有vocabulary.p檔
            print("Loading vocabulary from {}".format(self.vocab_path))
            return pickle.load(open(self.vocab_path,'rb'))
        print("Loading vocabulary from data")
        counter = Counter()
        # get vocabulary from dialogues
        for subset in ["train", "valid", "test"]:
            #在所有資料中
            for conversation in tqdm(self.conversation_data[subset]):
                #每一個對話
                for message in conversation["messages"]:
                    #每一句
                    pattern = re.compile(r'@(\d+)')
                    # remove movie Ids
                    text = tokenize(pattern.sub(" ", message["text"]))
                    #以空格斷字
                    counter.update([word.lower() for word in text])
                    #計算每個字出現次數

        # get vocabulary from movie names
        for movieId in self.db2name:
            tokenized_movie = tokenize(self.db2name[movieId])
            #取得所有電影名稱並斷字
            counter.update([word.lower() for word in tokenized_movie])
            #計算每個字出現次數
        
        #get vocabulary from movie reviews
        review_data = pd.read_csv(self.review_path)
        review_text = review_data['replaced_review_text'].tolist()
        for review in review_text:
            text = tokenize(review)
            counter.update([word.lower() for word in text])

        # Keep the most common words
        kept_vocab = counter.most_common(20000)
        #保留前20000單字
        vocab = [x[0] for x in kept_vocab]
        #被保留下的單字
        print("Vocab covers {} word instances over {}, that's about {} percent of the vocabulary.".format(
            sum([x[1] for x in kept_vocab]),
            #被保留的單字數量
            sum([counter[x] for x in counter]),
            #全部出現過的單字
            sum([x[1] for x in kept_vocab])/sum([counter[x] for x in counter])
            #計算保留比例
        ))
        vocab += ['<s>', '</s>', '<pad>', '<unk>', '\n']
        #加入需要label
        with open(self.vocab_path, 'wb') as f:
            pickle.dump(vocab, f)
            #寫入voc.p檔
        print("Saved vocabulary in {}".format(self.vocab_path))
        return vocab

    def _load_rw_rec_batch(self, subset, flatten_messages=True, cut_dialogues=-1):
        batch = {"convId": [], "movieName": [], "input_dist": [],"target": []}

        # get batch
        batch_data = self.re_rec_data[subset][self.batch_index[subset] * self.batch_size:
                                               (self.batch_index[subset] + 1) * self.batch_size]
        for i, example in enumerate(batch_data):
            batch['convId'].append(example['convID'])
            batch['movieName'].append(example['likemoviename'])
            for item in example["input_dist"]:
                batch["input_dist"].append(item)
            for item in example['target']:
                batch["target"].append(item)
        batch["target"] = Variable(torch.LongTensor(batch["target"]))  # (batch, 6)
        batch["input_dist"] = Variable(torch.FloatTensor(batch["input_dist"]))  # (batch, 6)
        return batch

    def dtext_to_ids(self, dialogue, max_utterance_len, max_conv_len):
        """
        replace with corresponding ids.
        Pad each utterance to max_utterance_len. And pad each conversation to max_conv_length
        :param dialogue: [[[word1, word2, ...]]]
        :param max_utterance_len:
        :param max_conv_len:
        :return: padded dialogue
        """
        dialogue = [[[self.token2id(w) for w in utterance] +
                     [self.word2id['<pad>']] * (max_utterance_len - len(utterance)) for utterance in conv] +
                    [[self.word2id['<pad>']] * max_utterance_len] * (max_conv_len - len(conv)) for conv in dialogue]
        return dialogue

    def replace_movies_in_tokenized(self, tokenized):
        """
        replace movieId tokens in a single tokenized message.
        Eventually compute the movie occurrences and the target with (global) movieIds
        :param tokenized:
        :return:
        """
        output_with_id = tokenized[:]
        occurrences = {}
        pattern = re.compile(r'^\d{5,6}$')
        index = 0
        while index < len(tokenized):
            word = tokenized[index]
            # Check if word corresponds to a movieId.
            if pattern.match(word) and int(word) in self.db2id and not self.db2name[int(word)].isspace():
                # get the global Id
                movieId = self.db2id[int(word)]
                # add movie to occurrence dict
                if movieId not in occurrences:
                    occurrences[movieId] = [0] * len(tokenized)
                # remove ID
                del tokenized[index]
                # put tokenized movie name instead. len(tokenized_movie) - 1 elements are added to tokenized.
                tokenized_movie = tokenize(self.id2name[movieId])
                tokenized[index:index] = tokenized_movie

                # update output_with_id: replace word with movieId repeated as many times as there are words in the
                # movie name. Add the size-of-vocabulary offset.
                output_with_id[index:index + 1] = [movieId + len(self.word2id)] * len(tokenized_movie)
                #print("output_tokenzied_movie:{}".format(output_with_id[index:index + 1]))
                # update occurrences
                if "movie_occurrences" in self.sources:
                    # extend the lists
                    for otherIds in occurrences:
                        # the zeros in occurrence lists can be appended at the end since all elements after index are 0
                        # occurrences[otherIds][index:index] = [0] * (len(tokenized_movie) - 1)
                        occurrences[otherIds] += [0] * (len(tokenized_movie) - 1)
                    # update list corresponding to the occurring movie
                    occurrences[movieId][index:index + len(tokenized_movie)] = [1] * len(tokenized_movie)

                # increment index
                index += len(tokenized_movie)

            else:
                # do nothing, and go to next word
                index += 1
        if "movie_occurrences" in self.sources:
            if "movieIds_in_target" in self.sources:
                return tokenized, output_with_id, occurrences
            return tokenized, tokenized, occurrences
        return tokenized
    
    def token2id(self, token):
        """
        :param token: string or movieId
        :return: corresponding ID
        """
        if token in self.word2id:
            return self.word2id[token]
        if isinstance(token, int):
            return token
        return self.word2id['<unk>']

    def truncate(self, dialogue, senders, movie_occurrences):
        # truncate conversations that have too many utterances
        if len(dialogue) > self.conversation_length_limit:
            dialogue = dialogue[:self.conversation_length_limit]
            # target = target[:self.conversation_length_limit]
            senders = senders[:self.conversation_length_limit]
            if "movie_occurrences" in self.sources:
                movie_occurrences = {
                    key: val[:self.conversation_length_limit] for key, val in movie_occurrences.items()
                }
        # truncate utterances that are too long
        for (i, utterance) in enumerate(dialogue):
            if len(utterance) > self.utterance_length_limit:
                dialogue[i] = dialogue[i][:self.utterance_length_limit]
                # target[i] = target[i][:self.utterance_length_limit]
                if "movie_occurrences" in self.sources:
                    for movieId, value in movie_occurrences.items():
                        value[i] = value[i][:self.utterance_length_limit]
        return dialogue, senders, movie_occurrences

    def set_word2id(self, word2id):
        self.word2id = word2id
        self.id2word = {id: word for (word, id) in self.word2id.items()}

        if self.process_at_instanciation:
            # pre-process dialogues
            self.conversation_data = {key: [self.extract_dialogue(conversation, flatten_messages=True)
                                            for conversation in val]
                                      for key, val in self.conversation_data.items()}

    def extract_dialogue(self, conversation, flatten_messages=True):
        """
        :param conversation: conversation dictionary. keys : 'conversationId', 'respondentQuestions', 'messages',
         'movieMentions', 'respondentWorkerId', 'initiatorWorkerId', 'initiatorQuestions'
         :param flatten_messages
         :return:
        """
        dialogue = []
        # target = []
        senders = []
        occurrences = None
        if "movie_occurrences" in self.sources:
            # initialize occurrences. Ignore empty movie names
            occurrences = {self.db2id[int(dbId)]: [] for dbId in conversation["movieMentions"]
                           if int(dbId) in self.db2name and not self.db2name[int(dbId)].isspace()}
        for message in conversation["messages"]:
            # role of the sender of message: 1 for seeker, -1 for recommender
            role = 1 if message["senderWorkerId"] == conversation["initiatorWorkerId"] else -1
            # remove "@" and add spaces around movie mentions to be sure to count them as single tokens
            # tokens that match /^\d{5,6}$/ are movie mentions
            pattern = re.compile(r'@(\d+)')
            message_text = pattern.sub(lambda m: " " + m.group(1) + " ", message["text"])
            text = tokenize(message_text)

            if "movie_occurrences" in self.sources:
                text, message_target, message_occurrences = self.replace_movies_in_tokenized(text)
            else:
                text = self.replace_movies_in_tokenized(text)
                message_target = text
                
            # if flatten messages, append message when the sender is the same as in the last message
            if flatten_messages and len(senders) > 0 and senders[-1] == role:
                dialogue[-1] += ["\n"] + text
                # target[-1] += ["\n"] + message_target
                if "movie_occurrences" in self.sources:
                    for movieId in occurrences:
                        if movieId in message_occurrences:
                            occurrences[movieId][-1] += [0] + message_occurrences[movieId]
                        else:
                            occurrences[movieId][-1] += [0] * (len(text) + 1)
            # otherwise finish the previous utterance and add the new utterance
            else:
                if len(senders) > 0:
                    dialogue[-1] += ['</s>']
                    # target[-1] += ['</s>', '</s>']
                    if "movie_occurrences" in self.sources:
                        for movieId in occurrences:
                            occurrences[movieId][-1] += [0]
                senders.append(role)
                dialogue.append(['<s>'] + text)
                # target.append(message_target)
                if "movie_occurrences" in self.sources:
                    for movieId in occurrences:
                        if movieId in message_occurrences:
                            occurrences[movieId].append([0] + message_occurrences[movieId])
                        else:
                            occurrences[movieId].append([0] * (len(text) + 1))
        # finish the last utterance
        dialogue[-1] += ['</s>']
        # target[-1] += ['</s>', '</s>']
        if "movie_occurrences" in self.sources:
            for movieId in occurrences:
                occurrences[movieId][-1] += [0]
        # dialogue, target, senders, occurrences = self.truncate(dialogue, target, senders, occurrences)
        dialogue,  senders, occurrences = self.truncate(dialogue, senders, occurrences)

        if "movie_occurrences" in self.sources:
            # return dialogue, target, senders, occurrences
            return dialogue,  senders, occurrences
        # return dialogue, target, senders, None
        return dialogue, senders, None

    def _load_dialogue_batch(self, subset, flatten_messages):
        batch = {"senders": [], "dialogue": [], "lengths": [], "target": [],"seeker_mentioned":[]}
        if "movie_occurrences" in self.sources:
            # movie occurrences: Array of dicts
            batch["movie_occurrences"] = []

        # get batch
        batch_data = self.conversation_data[subset][self.batch_index[subset] * self.batch_size:
                                                    (self.batch_index[subset] + 1) * self.batch_size]
        
        
        for i, conversation in enumerate(batch_data):
            if self.process_at_instanciation:
                dialogue, senders, movie_occurrences = conversation
            else:
                dialogue, senders, movie_occurrences = self.extract_dialogue(conversation,
                                                                                     flatten_messages=flatten_messages)
            #製作target data
            init_q = conversation["initiatorQuestions"]
            #存入seeker資料
            resp_q = conversation["respondentQuestions"]
            #存入recommender資料
            gen = (key for key in init_q if key in resp_q and not self.db2name[int(key)].isspace())
            gen2 = (key for key in init_q if key in resp_q and not self.db2name[int(key)].isspace())
            # get recommend success list
            recmovielist = []
            for key in gen:
                if ((resp_q[key]["suggested"] == 1) & (init_q[key]["suggested"] == 1) & (resp_q[key]["liked"] == 1) & (init_q[key]["liked"] == 1)):
                    recmovielist.append(self.db2id[int(key)])
            
            # get seeker mentioned movies
            seeker_mentioned = []
            # for key in gen2:
            #     if ((resp_q[key]["suggested"] == 1) | (init_q[key]["suggested"] == 1)):
            #         seeker_mentioned.append(self.db2id[int(key)])

            #build target 
            target_list = [0] * self.n_movies
            for mid in recmovielist:
                target_list[int(mid)] = 1 
                
            batch["lengths"].append([len(message) for message in dialogue])
            batch["dialogue"].append(dialogue)
            batch["senders"].append(senders)
            batch["target"].append(target_list)
            batch["seeker_mentioned"].append(seeker_mentioned)

            if "movie_occurrences" in self.sources:
                batch["movie_occurrences"].append(movie_occurrences)

        max_utterance_len = max([max(x) for x in batch["lengths"]])
        max_conv_len = max([len(conv) for conv in batch["dialogue"]])
        batch["conversation_lengths"] = np.array([len(x) for x in batch["lengths"]])
        # replace text with ids and pad sentences
        batch["lengths"] = np.array(
            [lengths + [0] * (max_conv_len - len(lengths)) for lengths in batch["lengths"]]
        )
        batch["dialogue"] = Variable(torch.LongTensor(
            self.dtext_to_ids(batch["dialogue"], max_utterance_len, max_conv_len)))
        batch["target"] = Variable(torch.LongTensor(batch["target"]))
        batch["seeker_mentioned"] = Variable(torch.LongTensor(batch["seeker_mentioned"]))
        batch["senders"] = Variable(torch.FloatTensor(
            [senders + [0] * (max_conv_len - len(senders)) for senders in batch["senders"]]))
        if "movie_occurrences" in self.sources:
            batch["movie_occurrences"] = [
                {key: [utterance + [0] * (max_utterance_len - len(utterance)) for utterance in value] +
                      [[0] * max_utterance_len] * (max_conv_len - len(value)) for key, value in conv.items()}
                for conv in batch["movie_occurrences"]
            ]
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
        if "dialogue" in self.sources:
            if self.word2id is None:
                raise ValueError("word2id is not set, cannot load batch")
            batch = self._load_dialogue_batch(subset, flatten_messages)

        self.batch_index[subset] = (self.batch_index[subset] + 1) % self.n_batches[subset]

        return batch
    
if __name__ == '__main__':

    #一次跑兩個model的比較
    rec_list = ["ori_rec","enhanced_rec"]
    #rec_list = ["ori_rec"]
    for rec in rec_list:
        if(rec == "ori_rec"):
            batch_loader = TestingBatchLoader(
                sources="dialogue movie_occurrences movieIds_in_target",
                batch_size=test_params.train_recommender_params['batch_size'] # 1
                )
            ori_rec = Ori_Recommender(train_vocab=batch_loader.train_vocabulary,
            n_movies=batch_loader.n_movies,
            params=test_params.recommender_params)

            load_model(ori_rec,'./models/recommender/MSE/checkpoint')
            batch_loader.set_word2id(ori_rec.encoder.word2id)
            ori_rec.test(batch_loader, subset="test")
        else:
            batch_loader = TestingBatchLoader(
                sources="dialogue movie_occurrences movieIds_in_target", # "sentiment_analysis movie_occurrences"
                batch_size=test_params.train_enhanced_recommender_params['batch_size'] # 1
                )
            enhanced_rec = Enhanced_Recommender(train_vocab=batch_loader.train_vocabulary,
            n_movies=batch_loader.n_movies,
            movie_rep_matrix=batch_loader.movie_rep_matrix, 
            params=test_params.enhanced_recommender_params)

            load_model(enhanced_rec,'./models/enhanced_recommender/final/experi/seq_initial/checkpoint')
            batch_loader.set_word2id(enhanced_rec.encoder.word2id)
            enhanced_rec.test(batch_loader, subset="test")
        

        
    
