#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author:lch02
@Time:  2017/12/26 18:23
@Description: 
"""
import config
import os
import nltk
import pickle

from nltk.classify import SklearnClassifier
from sklearn.svm import LinearSVC

__author__ = 'lch02'


def create_classifier(featx):

    pos_data = pickle.load(open(os.path.join(config.pkl_path, 'pos_reviews.pkl'), 'rb'))
    neg_data = pickle.load(open(os.path.join(config.pkl_path, 'neg_reviews.pkl'), 'rb'))

    pos_test_data = pickle.load(open(os.path.join(config.pkl_path, 'test_pos_reviews.pkl'), 'rb'))
    neg_test_data = pickle.load(open(os.path.join(config.pkl_path, 'test_neg_reviews.pkl'), 'rb'))

    print len(pos_data), '---++---', len(neg_data)
    pos_features = [(featx(w_lst), 'pos') for w_lst in pos_data]
    neg_features = [(featx(w_lst), 'neg') for w_lst in neg_data]

    pos_test_features = [(featx(w_lst), 'pos') for w_lst in pos_test_data]
    neg_test_features = [(featx(w_lst), 'neg') for w_lst in neg_test_data]

    pos_features.extend(neg_features)
    train_set = pos_features

    pos_test_features.extend(neg_test_features)
    test_set = pos_test_features

    print train_set is None, '---train_set----', len(train_set)
    print test_set is None, '-----test_set--', len(test_set)

    """
    训练两个分类器
    """
    nb_classifier = nltk.NaiveBayesClassifier.train(train_set)
    nba = nltk.classify.accuracy(nb_classifier, test_set)
    print "NBayes accuracy is %.7f" % nba  # 86.78%

    svm_classifier = SklearnClassifier(LinearSVC()).train(train_set)
    svmm = nltk.classify.accuracy(svm_classifier, test_set)
    print "svm_classifier accuracy is %.7f" % svmm  # 89.124%

    """
    保存准确率更大的那个模型
    """
    classifier_pkl = os.path.join(config.pkl_path, 'my_classifier.pkl')  # 消极语料
    with open(classifier_pkl, 'wb') as f:
        if nba > svmm:
            pickle.dump(nb_classifier, f)
            print 'NBayes'
        else:
            pickle.dump(svm_classifier, f)
            print 'SVM'

    print 'done!'


def get_model():
    with open(os.path.join(config.pkl_path, 'my_classifier.pkl'), 'rb') as f:
        classifier = pickle.load(f)
    return classifier
