# -*- coding: utf-8 -*-
__author__ = 'xincz'

import numpy as np
from numpy.linalg import solve
from scipy.optimize import approx_fprime

from utils import common, findMin


# Logistic Regression
class LogReg:

    def __init__(self, verbose=1, maxEvals=100):
        self.verbose = verbose
        self.maxEvals = maxEvals
        self.bias = True
        self.w = None

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

        # Initial guess
        self.w = np.zeros(d)
        common.check_gradient(self, X, y)
        (self.w, f) = findMin.findMin(self.funObj, self.w,
                                      self.maxEvals, X, y, verbose=self.verbose)

    def predict(self, X):
        return np.sign(X@self.w)


# Logistic Regression with l2-regularization
class LogRegL2(LogReg):

    def __init__(self, verbose=1, lammy=1.0, maxEvals=100):
        self.verbose = verbose
        self.lammy = lammy
        self.maxEvals = maxEvals
        self.bias = True

    def funObj(self, w, X, y):
        yXw = y * X.dot(w)

        # Calculate the function value
        f = np.sum(np.log(1. + np.exp(-yXw))) + self.lammy/2. * w.dot(w)

        # Calculate the gradient value
        res = - y / (1. + np.exp(yXw))
        g = X.T.dot(res) + self.lammy * w

        return f, g

    def fit(self, X, y):
        n, d = X.shape

        # Initial guess
        self.w = np.zeros(d)
        common.check_gradient(self, X, y)
        (self.w, f) = findMin.findMin(self.funObj, self.w,
                                      self.maxEvals, X, y, verbose=self.verbose)

    def predict(self, X):
        return np.sign(X@self.w)


# Logistic Regression with l1-regularization
class LogRegL1(LogReg):

    def __init__(self, L1_lambda=1.0, verbose=1, maxEvals=100):
        self.verbose = verbose
        self.L1_lambda = L1_lambda
        self.maxEvals = maxEvals

    def fit(self, X, y):
        n, d = X.shape

        # Initial guess
        self.w = np.zeros(d)
        common.check_gradient(self, X, y)
        (self.w, f) = findMin.findMinL1(self.funObj, self.w, self.L1_lambda,
                                        self.maxEvals, X, y, verbose=self.verbose)


# Logistic Regression with l0-regularization
class LogRegL0(LogReg):

    def __init__(self, L0_lambda=1.0, verbose=2, maxEvals=400):
        self.verbose = verbose
        self.L0_lambda = L0_lambda
        self.maxEvals = maxEvals

    def fit(self, X, y):
        n, d = X.shape
        minimize = lambda ind: findMin.findMin(self.funObj, np.zeros(len(ind)),
                                               self.maxEvals, X[:, ind], y, verbose=0)
        selected = set()
        selected.add(0)
        minLoss = np.inf
        oldLoss = 0
        bestFeature = -1

        while minLoss != oldLoss:
            oldLoss = minLoss
            print("Epoch %d " % len(selected))
            print("Selected feature: %d" % (bestFeature))
            print("Min Loss: %.3f\n" % minLoss)

            for i in range(d):  # find the best feature
                if i in selected:
                    continue

                selected_new = selected | {i}  # tentatively add feature "i" to the selected set
                # then compute the loss and update the minLoss/bestFeature
                w, f = minimize(list(selected_new))
                # Add the L0 term
                f += self.L0_lambda * len(selected_new)

                if f <= minLoss:
                    minLoss = f
                    bestFeature = i

            selected.add(bestFeature)

        self.w = np.zeros(d)
        self.w[list(selected)], _ = minimize(list(selected))
