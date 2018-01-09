#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author:lch02
@Time:  2017/12/25 14:46
@Description:
"""
import re

from nltk import regexp_tokenize, pos_tag, WordNetLemmatizer
from nltk.corpus import stopwords
from config import *
from multiprocessing import Pool

__author__ = 'lch02'


"""
将test中的数据导出
"""
def export_test_data():
    pool = Pool()
    files = ['neg', 'pos']
    for i in range(2):
        pool.apply_async(deal_test_doc, args=(i, files[i]))
    pool.close()
    pool.join()
    print 'import test'

def deal_test_doc(cat, fn):
    file_path = os.path.join(test_path, fn)
    files = os.listdir(file_path)
    neg = []
    pos = []
    if cat == 0:
        for fl in files:
            with open(os.path.join(file_path, fl), 'rb') as reader:
                txt = reader.read().decode('utf-8')
            if txt is not None:
                content = text_parse(txt)
                if len(content) == 0:
                    continue
                elif len(content) != 0:
                    neg.append(content)
        neg_file = os.path.join(pkl_path, 'test_neg_reviews.pkl')  # 消极语料
        with open(neg_file, 'wb') as f:
            pickle.dump(neg, f)
    elif cat == 1:
        for fl in files:
            with open(os.path.join(file_path, fl), 'rb') as reader:
                txt = reader.read().decode('utf-8')
            if txt is not None:
                content = text_parse(txt)
                if len(content) == 0:
                    continue
                elif len(content) != 0:
                    pos.append(content)
        pos_file = os.path.join(pkl_path, 'test_pos_reviews.pkl')  # 积极语料
        with open(pos_file, 'wb') as f:
            pickle.dump(pos, f)

"""
将train中的数据导出
"""
def export_train_data():
    pool = Pool()
    files = ['neg', 'pos', 'other']
    for i in range(3):
        pool.apply_async(deal_doc, args=(i, files[i]))
    pool.close()
    pool.join()
    print 'import train'

def deal_doc(cat, fn):
    file_path = os.path.join(train_path, fn)
    files = os.listdir(file_path)
    neg = []
    pos = []
    feat = []
    if cat == 0:
        for fl in files:
            with open(os.path.join(file_path, fl), 'rb') as reader:
                txt = reader.read().decode('utf-8')
            if txt is not None:
                content = text_parse(txt)
                if len(content) == 0:
                    continue
                elif len(content) != 0:
                    neg.append(content)
        neg_file = os.path.join(pkl_path, 'neg_reviews.pkl')  # 消极语料
        with open(neg_file, 'wb') as f:
            pickle.dump(neg, f)
    elif cat == 1:
        for fl in files:
            with open(os.path.join(file_path, fl), 'rb') as reader:
                txt = reader.read().decode('utf-8')
            if txt is not None:
                content = text_parse(txt)
                if len(content) == 0:
                    continue
                elif len(content) != 0:
                    pos.append(content)
        pos_file = os.path.join(pkl_path, 'pos_reviews.pkl')  # 积极语料
        with open(pos_file, 'wb') as f:
            pickle.dump(pos, f)
    elif cat == 2:
        for fl in files:
            with open(os.path.join(file_path, fl), 'rb') as reader:
                lines = reader.readlines()

                for line in lines:
                    line = line.strip('\n')
                    feat.append(line.decode('utf-8'))
        feat_file = os.path.join(pkl_path, 'feats.pkl')  # 积极语料
        with open(feat_file, 'wb') as f:
            pickle.dump(feat, f)

"""
文本处理：取词、去停用词等
"""
def text_parse(input_text, language='en'):
    lemmatizer = WordNetLemmatizer()  # 词形还原
    sentence = input_text.strip().lower()
    sentence = re.sub(r'@\s*[\w]+ | ?#[\w]+ | ?&[\w]+; | ?[^\x00-\xFF]+', '', sentence)
    special_tag = set(
        ['.', ',', '#', '!', '(', ')', '*', '`', ':', '"', '‘', '’', '“', '”', '@', '：', '^', '/', ']', '[', ';', '=', '_'])
    pattern = r""" (?x)(?:[a-z]\.)+ 
                  | \d+(?:\.\d+)?%?\w+
                  | \w+(?:[-']\w+)*"""

    tag_list = set(['TO', 'RB', 'RBR', 'RBRS', 'UH', 'WDT', 'WP', 'WP$', 'WRB', 'SYM', 'RP', 'PRP', 'PRP$', 'CD', 'POS', ':'])
    word_list = regexp_tokenize(sentence, pattern)
    if language == 'en':
        filter_word = [w for w in word_list if w not in stopwords.words('english') and w not in special_tag]  # 去停用词和特殊标点符号
    word_tag = pos_tag(filter_word, tagset=None, lang=language)  # 词性标注，返回标记列表[('Codeine', 'NNP'), ('15mg', 'CD')
    res_word_list = []
    for i in range(0, len(word_tag)):  # 去掉副词、介词、小品词、疑问词、代词、人称代词、所有格代名词等
        if word_tag[i][1] in tag_list:
            continue
        else:
            word = lemmatizer.lemmatize(word_tag[i][0])
            res_word_list.append(word)
    return res_word_list
