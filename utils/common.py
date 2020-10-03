# -*- coding: utf-8 -*-
__author__ = 'xincz'

import numpy as np
import scipy.sparse
from scipy import stats
from scipy.optimize import approx_fprime


def mode(y):
    """
    Computes the element with the maximum count
    :param y: an input numpy array
    :return: the element with the maximum count
    """
    if len(y) == 0:
        return -1
    else:
        return stats.mode(y.flatten())[0][0]


def euclidean_dist_squared(X, Xtest):
    """
    Computes the Euclidean distance between rows of 'X' and rows of 'Xtest'
    :param X: an N by D numpy array
    :param Xtest: an T by D numpy array
    :return: an array of size N by T containing the pairwise squared Euclidean distances.
    """
    return np.sum(X**2, axis=1)[:, None] + np.sum(Xtest**2, axis=1)[None] - 2 * np.dot(X, Xtest.T)


def check_gradient(model, X, y):
    # This checks that the gradient implementation is correct
    w = np.random.rand(model.w.size)
    f, g = model.funObj(w, X, y)

    # Check the gradient
    estimated_gradient = approx_fprime(w,
                                       lambda w: model.funObj(w,X,y)[0],
                                       epsilon=1e-6)

    implemented_gradient = model.funObj(w, X, y)[1]

    if np.max(np.abs(estimated_gradient - implemented_gradient) > 1e-4):
        raise Exception('User and numerical derivatives differ:\n%s\n%s' %
             (estimated_gradient[:5], implemented_gradient[:5]))
    else:
        print('User and numerical derivatives agree.')


def classification_error(y, yhat):
    return np.mean(y != yhat)


def dijkstra(G, i=None, j=None):
    """
    Computes shortest distance between all pairs of nodes given an adjacency
    matrix G, where G[i,j]=0 implies there is no edge from i to j.
    :param G: an N by N numpy array
    :param i: node i
    :param j: node j
    :return: shortest distance
    """
    dist = scipy.sparse.csgraph.dijkstra(G, directed=False)
    if i is not None and j is not None:
        return dist[i,j]
    else:
        return dist
