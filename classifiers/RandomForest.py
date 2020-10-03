# -*- coding: utf-8 -*-
__author__ = 'xincz'

import numpy as np

from utils import common
from classifiers.RandomTree import RandomTree


class RandomForest:

    def __init__(self, num_trees, max_depth):
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.random_trees = []

    def fit(self, X, y):
        for k in range(self.num_trees):
            # get a bootstrap sample
            model = RandomTree(self.max_depth)
            model.fit(X, y)
            self.random_trees.append(model)

    def predict(self, X):
        N = X.shape[0]
        all_preds = np.zeros([N, self.num_trees])

        j = 0
        for model in self.random_trees:
            all_preds[:, j] = model.predict(X)
            j += 1

        yhat = np.zeros(N)
        for i in range(N):
            yhat[i] = common.mode(all_preds[i, :])

        return yhat
