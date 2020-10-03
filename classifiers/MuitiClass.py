# -*- coding: utf-8 -*-
__author__ = 'xincz'

import numpy as np
from numpy.linalg import solve
from scipy.optimize import approx_fprime

from utils import common, findMin


class LeastSquaresClassifier:

    def __init__(self):
        self.n_classes = None
        self.W = None

    def fit(self, X, y):
        n, d = X.shape
        self.n_classes = np.unique(y).size

        # Initial guess
        self.W = np.zeros((self.n_classes,d))

        for i in range(self.n_classes):
            ytmp = y.copy().astype(float)
            ytmp[y == i] = 1
            ytmp[y != i] = -1

            # solve the normal equations
            # with a bit of regularization for numerical reasons
            self.W[i] = np.linalg.solve(X.T@X+0.0001*np.eye(d), X.T@ytmp)

    def predict(self, X):
        return np.argmax(X@self.W.T, axis=1)


class LogLinearClassifier:
    def __init__(self, verbose=0, maxEvals=100):
        self.verbose = verbose
        self.maxEvals = maxEvals
        self.bias = True
        self.n_classes = None
        self.w = None
        self.W = None

    def funObj(self, w, X, y):
        yXw = y * X.dot(w)

        # Calculate the function value
        f = np.sum(np.log(1. + np.exp(-yXw)))

        # Calculate the gradient value
        res = - y / (1. + np.exp(yXw))
        g = X.T.dot(res)

        return f, g

    def fit(self, X, y):
        n, d = X.shape
        self.n_classes = np.unique(y).size

        # Initial guess
        self.W = np.zeros((self.n_classes, d))

        for i in range(self.n_classes):
            ytmp = y.copy().astype(float)
            ytmp[y == i] = 1
            ytmp[y != i] = -1

            self.w = self.W[i]
            common.check_gradient(self, X, ytmp)
            (self.W[i], f) = findMin.findMin(self.funObj, self.W[i], self.maxEvals,
                                             X, ytmp, verbose=self.verbose)

    def predict(self, X):
        return np.argmax(X @ self.W.T, axis=1)  # result: n by 1 vector of classes


class SoftmaxClassifier:
    def __init__(self, verbose=0, maxEvals=100):
        self.verbose = verbose
        self.maxEvals = maxEvals
        self.bias = True
        self.n_classes = None
        self.W = None
        self.w = None

    def funObj(self, w, X, y):  # w is of length kd
        n, d = X.shape
        k = np.unique(y).size
        W = np.reshape(w, (k, d))  # W is k by d matrix

        # Calculate the function value
        f = 0
        for i in range(n):
            f += np.log(np.sum(np.exp(X[i] @ W.T)))
            f -= X[i] @ W[y[i]].T

        # Calculate the gradient value
        g = np.zeros((k, d))
        for i in range(n):
            softmax_all = np.exp(W @ X[i].T)  # k by 1
            for c in range(k):
                softmax = softmax_all[c] / np.sum(softmax_all)  # probability
                g[c] += X[i] * (softmax - int(y[i] == c))  # 1 by d

        return f, g.flatten()  # returns as kd

    def fit(self, X, y):
        n, d = X.shape
        self.n_classes = np.unique(y).size
        self.W = np.zeros((self.n_classes, d))
        self.w = self.W.flatten()
        common.check_gradient(self, X, y)

        # Optimizer takes kd and returns kd
        (self.w, f) = findMin.findMin(self.funObj, self.w, self.maxEvals,
                                      X, y, verbose=self.verbose)
        self.W = self.w.reshape((self.n_classes, d))

    def predict(self, X):
        return np.argmax(X @ self.W.T, axis=1)  # result: t by 1 vector of classes
