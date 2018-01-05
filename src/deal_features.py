#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author:lch02
@Time:  2017/12/26 9:51
@Description: 
"""
import itertools
import os
import pickle
import config
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from config import test_path
from nltk.probability import FreqDist, ConditionalFreqDist
__author__ = 'lch02'


"""
选择贡献最大的特征
"""
def get_best_words(scores_dict, threshold=10000):
    best = sorted(scores_dict.iteritems(), key=lambda (word, score): score, reverse=True)[:threshold]   # 从大到小排列,选择前10000个
    best_words = set([w for w, s in best])
    return best_words


"""  
最有信息量的单个词作为特征
"""
def best_words_features(words):
    if len(config.best_words) == 0:
        config.best_words = pickle.load(open(os.path.join(config.pkl_path, 'feats.pkl'), 'rb'))
    lst = []
    for word in words:
        if word in config.best_words:
            lst.append((word, True))
        else:
            lst.append((word, False))
    return dict(lst)


"""
把所有词和双词搭配一起作为特征
"""
def best_bigram_words_features(words, score_fn=BigramAssocMeasures.chi_sq, n=1500):
    try:
        bigram_finder = BigramCollocationFinder.from_words(words)
        bigrams = bigram_finder.nbest(score_fn, n)
    except ZeroDivisionError:
        words.append(' ')
        bigram_finder = BigramCollocationFinder.from_words(words)
        bigrams = bigram_finder.nbest(score_fn, n)
    d = dict([(bigram, True) for bigram in bigrams])
    d.update(best_words_features(words))
    return d
