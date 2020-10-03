# -*- coding: utf-8 -*-
__author__ = 'xincz'

import numpy as np

from utils.common import euclidean_dist_squared


class KMeans:

    def __init__(self, k):
        self.k = k

    def fit(self, X):
        N, D = X.shape
        y = np.ones(N)

        # Initialize the means
        means = np.zeros((self.k, D))
        for kk in range(self.k):
            i = np.random.randint(N)
            means[kk] = X[i]

        # Minimization of cost function
        while True:
            y_old = y

            # Compute euclidean distance to each mean
            dist2 = euclidean_dist_squared(X, means)  # N by k
            dist2[np.isnan(dist2)] = np.inf
            y = np.argmin(dist2, axis=1)  # for each example find the closest cluster, update y

            # Update means
            for kk in range(self.k):
                # don't update the mean if no examples are assigned to it
                if np.any(y == kk):
                    means[kk] = X[y == kk].mean(axis=0)

            changes = np.sum(y != y_old)

            if changes == 0:
                break

        self.means = means

    def predict(self, X):
        means = self.means
        dist2 = euclidean_dist_squared(X, means)
        dist2[np.isnan(dist2)] = np.inf
        return np.argmin(dist2, axis=1)

    def error(self, X):
        dist2 = euclidean_dist_squared(X, self.means)  # N by k
        dist2[np.isnan(dist2)] = np.inf
        y = np.argmin(dist2, axis=1)  # N
        return np.sum(np.linalg.norm(X-self.means[y], axis=1)**2)  # return the cost
