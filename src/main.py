#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author:lch02
@Time:  2017/12/25 14:32
@Description: 
"""

import socket
import json
from deal_features import *
from my_classifier import *
from src.deal_files import *

__author__ = 'lch02'

def main():
    classifier = get_model()
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('127.0.0.1', 9797))
    s.listen(10)
    while True:
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
            continue
        deal = best_words_features(text_parse(data))
        res = classifier.classify(deal)
        pdist = classifier.prob_classify(deal)
        p = pdist.prob('pos')
        res_json = {'class': res, 'prob': p}
        res_json = json.dumps(res_json)
        sock.send(res_json)


if __name__ == '__main__':
    # main()
    # export_train_data()
    # export_test_data()

    config.best_words = pickle.load(open(os.path.join(config.pkl_path, 'feats.pkl'), 'rb'))
    create_classifier(best_bigram_words_features)