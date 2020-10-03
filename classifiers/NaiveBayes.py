# -*- coding: utf-8 -*-
__author__ = 'xincz'

import numpy as np


class NaiveBayes:
    """
    Assumes the feature are binary.
    Also assumes the labels go from 0,1,...C-1.
    """
    def __init__(self, num_classes, beta=0):
        self.num_classes = num_classes
        self.beta = beta
        self.p_xy = None
        self.p_y = None

    def fit(self, X, y):
        N, D = X.shape
        C = self.num_classes

        # Compute the probability of each class i.e p(y==c)
        counts = np.bincount(y)
        p_y = counts / N

        # Compute the conditional probabilities i.e.
        p_xy = np.zeros([D, C])
        for i in range(C):
            p_xy[:, i] = (np.sum(X[np.where(y == i)[0]], axis=0) + self.beta) / (counts[i] + self.beta*C)

        self.p_y = p_y
        self.p_xy = p_xy

    def predict(self, X):
        N, D = X.shape
        C = self.num_classes
        p_xy = self.p_xy
        p_y = self.p_y

        y_pred = np.zeros(N)
        for n in range(N):
            # initialize with the p(y) terms
            probs = p_y.copy()
            for d in range(D):
                if X[n, d] != 0:
                    probs *= p_xy[d, :]
                else:
                    probs *= (1-p_xy[d, :])
            y_pred[n] = np.argmax(probs)

        return y_pred
