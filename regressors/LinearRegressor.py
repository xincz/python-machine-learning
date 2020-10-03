# -*- coding: utf-8 -*-
__author__ = 'xincz'

import numpy as np
from numpy.linalg import solve
from scipy.optimize import approx_fprime

from utils.findMin import findMin
import utils


# Ordinary Least Squares
class LeastSquares:

    def __init__(self):
        self.w = None

    def fit(self, X, y):
        self.w = solve(X.T@X, X.T@y)

    def predict(self, X):
        return X@self.w


# Least squares where each sample point X has a weight associated with it.
class WeightedLeastSquares(LeastSquares):
    # inherits the predict() function from LeastSquares
    def fit(self, X, y, z):
        self.w = solve(X.T@(X*z), X.T@(y*z))


class LinearModelGradient(LeastSquares):

    def fit(self, X, y):
        n, d = X.shape

        # Initial guess
        self.w = np.zeros((d, 1))

        # check the gradient
        estimated_gradient = approx_fprime(self.w, lambda w: self.funObj(w, X, y)[0], epsilon=1e-6)
        implemented_gradient = self.funObj(self.w, X, y)[1]
        if np.max(np.abs(estimated_gradient - implemented_gradient) > 1e-4):
            print('User and numerical derivatives differ: %s vs. %s' % (estimated_gradient, implemented_gradient))
        else:
            print('User and numerical derivatives agree.')

        self.w, f = findMin(self.funObj, self.w, 100, X, y)

    def funObj(self, w, X, y):
        # Calculate the function value
        # f = 0.5*np.sum((X@w - y)**2)
        f = np.sum(np.log(np.exp(X @ w - y) + np.exp(y - X @ w)))

        # Calculate the gradient value
        # g = X.T@(X@w-y)
        n, d = X.shape
        g = np.zeros((d, 1))

        for j in range(d):
            s = 0
            for i in range(n):
                a = np.exp(w.T * X[i, j] - y[i]) - np.exp(y[i] - w.T * X[i, j])
                b = np.exp(w.T * X[i, j] - y[i]) + np.exp(y[i] - w.T * X[i, j])
                s += X[i, j] * a / b
            g[j] = s

        return f, g


# Least Squares with a bias added
class LeastSquaresBias:

    def __init__(self):
        self.v = None

    def fit(self, X, y):
        Z = np.append(np.ones(X.shape), X, axis=1)
        self.v = solve(Z.T@Z, Z.T@y)

    def predict(self, X):
        Z = np.append(np.ones(X.shape), X, axis=1)
        return Z@self.v


# Least Squares with polynomial basis
class LeastSquaresPoly:

    def __init__(self, p):
        self.leastSquares = LeastSquares()
        self.p = p

    def fit(self,X,y):
        Z = self._polyBasis(X)
        self.v = solve(Z.T@Z, Z.T@y)

    def predict(self, X):
        Z = self._polyBasis(X)
        return Z@self.v

    def _polyBasis(self, X):
        """
        Helper function to transform any matrix X into the polynomial
        basis defined by this class at initialization.
        Returns the matrix Z that is the polynomial basis of X.
        """
        # Intercept
        Z = np.ones(X.shape)

        for i in range(1, self.p+1):
            Z = np.append(Z, np.power(X, i), axis=1)

        return Z
