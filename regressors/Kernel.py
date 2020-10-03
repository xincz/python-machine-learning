# -*- coding: utf-8 -*-
__author__ = 'xincz'

import numpy as np
from numpy.linalg import solve
from scipy.optimize import approx_fprime

from utils import common
from utils import findMin


def log_1_plus_exp_safe(x):
    """
    Compute log(1+exp(x)) in a numerically safe way,
    avoiding overflow/underflow issues.
    """
    out = np.log(1+np.exp(x))
    out[x > 100] = x[x > 100]
    out[x < -100] = np.exp(x[x < -100])
    return out


def kernel_RBF(X1, X2, sigma=1):
    X1_norm = np.sum(X1**2, axis=1)
    X2_norm = np.sum(X2**2, axis=1)
    return np.exp(-(X1_norm[:, None] + X2_norm[None, :] - 2*X1@X2.T)/(2*np.power(sigma, 2)))


def kernel_poly(X1, X2, p=2):
    return np.power((1 + X1@X2.T), p)


def kernel_linear(X1, X2):
    return X1@X2.T


class KernelLogRegL2:

    def __init__(self, lammy=1.0, verbose=0, maxEvals=100, kernel_fun=kernel_RBF, **kernel_args):
        self.verbose = verbose
        self.lammy = lammy
        self.maxEvals = maxEvals
        self.kernel_fun = kernel_fun
        self.kernel_args = kernel_args

    def funObj(self, u, K, y):
        yKu = y * (K@u)

        # Calculate the function value
        f = np.sum(np.log(1. + np.exp(-yKu)))
        f = np.sum(log_1_plus_exp_safe(-yKu))

        # Add L2 regularization
        f += 0.5 * self.lammy * u.T@K@u

        # Calculate the gradient value
        res = - y / (1. + np.exp(yKu))
        g = (K.T@res) + self.lammy * K@u
        return f, g

    def fit(self, X, y):
        n, d = X.shape
        self.X = X
        K = self.kernel_fun(X,X, **self.kernel_args)
        common.check_gradient(self, K, y, n, verbose=self.verbose)
        self.u, f = findMin.findMin(self.funObj, np.zeros(n), self.maxEvals, K, y, verbose=self.verbose)

    def predict(self, Xtest):
        Ktest = self.kernel_fun(Xtest, self.X, **self.kernel_args)
        return np.sign(Ktest@self.u)
