#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author:lch02
@Time:  2017/12/25 14:32
@Description: 
"""

import socket
import json
import os
from deal_features import *
from my_classifier import *
from src.deal_files import *

__author__ = 'lch02'


def main():
    classifier = get_model()
    print 'load success!'
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('127.0.0.1', 7004))
    s.listen(10)
    while True:
        try:
            sock, addr = s.accept()
            data = sock.recv(102400)
            data = data.decode('utf-8').encode('utf-8')
            if data is 'auto':
                export_train_data()
                export_test_data()
                if len(config.best_words) == 0:
                    config.best_words = pickle.load(open(os.path.join(config.pkl_path, 'feats.pkl'), 'rb'))
                create_classifier(best_bigram_words_features)
                sock.send("complete auto!")
                sock.close()
                continue
            deal = best_words_features(text_parse(data))
            res = classifier.classify(deal)
            if res == 'pos':
                p = 1.0
            else:
                p = 0.0
            res_json = json.dumps(p).encode('utf-8')
            sock.send(res_json)
            sock.close()
        except KeyboardInterrupt:
            print 'Exit'
            exit()
        except:
            print 'Error'


def check_dir():
    if not os.path.exists(pkl_path):
        os.makedirs(pkl_path)


if __name__ == '__main__':
    check_dir()
    main()
    """
    export_train_data()
    export_test_data()
    config.best_words = pickle.load(open(os.path.join(config.pkl_path, 'feats.pkl'), 'rb'))
    create_classifier(best_bigram_words_features)
    classifier = get_model()
    data = r'Once again Mr. Costner has dragged out a movie for far longer than necessary. Aside from the terrific sea rescue sequences, of which there are very few I just did not care about any of the characters. Most of us have ghosts in the closet, and Costner\'s character are realized early on, and then forgotten until much later, by which time I did not care. The character we should really care about is a very cocky, overconfident Ashton Kutcher. The problem is he comes off as kid who thinks he\'s better than anyone else around him and shows no signs of a cluttered closet. His only obstacle appears to be winning over Costner. Finally when we are well past the half way point of this stinker, Costner tells us all about Kutcher\'s ghosts. We are told why Kutcher is driven to be the best with no prior inkling or foreshadowing. No magic here, it was all I could do to keep from turning it off an hour in.'
    deal = best_words_features(text_parse(data))
    res = classifier.classify(deal)
    if res == 'pos':
        p = 1.0
    else:
        p = 0.0
    res_json = json.dumps(p).encode('utf-8')
    print p, '==='
    """
