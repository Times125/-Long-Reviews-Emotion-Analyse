#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author:lch02
@Time:  2018/1/5 20:29
@Description: 
"""
import numpy as np
from sklearn.svm import LinearSVC

__author__ = 'lch02'

class LinearSVC_proba(LinearSVC):

    def __platt_func(self, x):
        return 1/(1+np.exp(-x))

    def predict_proba(self, X):
        f = np.vectorize(self.__platt_func)
        raw_predictions = self.decision_function(X)
        platt_predictions = f(raw_predictions)
        probs = platt_predictions / platt_predictions.sum(axis=1)[:, None]
        return probs