import os
import torch
from torch.autograd import Variable
import numpy as np
import nltk
import time

movie_list = []

def permute(items):
    length = len(items)
    if length <= 10:
        def inner(ix=[]):
            do_yield = len(ix) == length - 1
            for i in range(0, length):
                if i in ix: #avoid duplicates
                    continue
                if do_yield:
                    yield tuple([items[y] for y in ix + [i]])
                else:
                    for p in inner(ix + [i]):
                        yield p
        return inner()
    else:
        print("too much for permutation: {}".format(length))
        return ""
        


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_model(model, resume, verbose=True):
    if not os.path.isfile(resume):
        raise ValueError("no checkpoint found at '{}'".format(resume))
    if model.cuda_available:
        checkpoint = torch.load(resume)
    else:
        checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    if verbose:
        print("=> loaded checkpoint '{}' (epoch {} with best loss {})"
              .format(resume, checkpoint['epoch'], checkpoint['best_loss']))


def sort_for_packed_sequence(lengths, cuda=False):
    """

    :param lengths: 1D array of lengths
    :return: sorted_lengths (lengths in descending order,
    sorted_idx (indices to sort),
    rev (indices to retrieve original order)
    """
    sorted_idx = np.argsort(lengths)[::-1]  # idx to sort by length
    sorted_lengths = lengths[sorted_idx]
    rev = np.argsort(sorted_idx)  # idx to retrieve original order

    tt = torch.cuda.LongTensor if cuda else torch.LongTensor
    sorted_idx = Variable(tt(sorted_idx.copy()))
    rev = Variable(tt(rev.copy()))
    return sorted_lengths, sorted_idx, rev


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

def tokenize_review(message):
    #句子斷字 轉小寫
    """
    Text processing: Sentence tokenize, then concatenate the word_tokenize of each sentence. Then lower.
    :param message:
    :return:
    """
    #sentences = nltk.sent_tokenize(message)
    tokenized = []
    for sentence in message:
        tmp =[word.lower() for word in nltk.word_tokenize(str(sentence))]
        tokenized.append(tmp)
    return tokenized


def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        print("Executed {} in {} s".format(func.__name__, time.time() - start_time))
        return result
    return wrapper
