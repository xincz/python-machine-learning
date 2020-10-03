# -*- coding: utf-8 -*-
__author__ = 'xincz'

import numpy as np
from scipy import stats

from utils import common


class KNN:

    def __init__(self, k):
        self.k = k
        self.X = None
        self.y = None

    def fit(self, X, y):
        # Just store the training data
        self.X = X
        self.y = y

    def predict(self, Xtest):
        T, D = Xtest.shape
        yhat = np.zeros(T)
        sq_dists = common.euclidean_dist_squared(self.X, Xtest)  # N by T

        for j in range(T):
            yhat[j] = np.bincount(self.y[np.argsort(sq_dists[:, j])[:self.k]]).argmax()

        return yhat
