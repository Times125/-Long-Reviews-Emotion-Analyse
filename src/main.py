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
            """
            if data is 'auto':
                export_train_data()
                export_test_data()
                if len(config.best_words) == 0:
                    config.best_words = pickle.load(open(os.path.join(config.pkl_path, 'feats.pkl'), 'rb'))
                create_classifier(best_bigram_words_features)
                sock.send("complete auto!")
                sock.close()
                continue
            """
            deal = best_words_features(text_parse(data))
            res = classifier.classify(deal)
            print res
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
    """

