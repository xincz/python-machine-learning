# -*- coding: utf-8 -*-
__author__ = 'xincz'

import numpy as np

from utils import common
from utils import findMin


# helper functions to transform between one big vector of weights
# and a list of layer parameters of the form (W,b)
def _flatten_weights(weights):
    return np.concatenate([w.flatten() for w in sum(weights, ())])


def _unflatten_weights(weights_flat, layer_sizes):
    weights = list()
    counter = 0
    for i in range(len(layer_sizes) - 1):
        W_size = layer_sizes[i + 1] * layer_sizes[i]
        b_size = layer_sizes[i + 1]

        W = np.reshape(weights_flat[counter:counter + W_size], (layer_sizes[i + 1], layer_sizes[i]))
        counter += W_size

        b = weights_flat[counter:counter + b_size][None]
        counter += b_size

        weights.append((W, b))
    return weights


def _log_sum_exp(Z):
    Z_max = np.max(Z, axis=1)
    # per-column max
    return Z_max + np.log(np.sum(np.exp(Z - Z_max[:, None]), axis=1))


class NeuralNet:
    # uses sigmoid nonlinearity
    def __init__(self, hidden_layer_sizes, lammy=1, max_iter=100):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.lammy = lammy
        self.max_iter = max_iter
        self.layer_sizes = None
        self.classification = None
        self.weights = None

    def funObj(self, weights_flat, X, y):
        weights = _unflatten_weights(weights_flat, self.layer_sizes)

        activations = [X]
        for W, b in weights:
            Z = X @ W.T + b
            X = 1 / (1 + np.exp(-Z))
            activations.append(X)

        yhat = Z

        if self.classification:
            tmp = np.sum(np.exp(yhat), axis=1)
            # f = -np.sum(yhat[y.astype(bool)] - np.log(tmp))
            f = -np.sum(yhat[y.astype(bool)] - _log_sum_exp(yhat))
            grad = np.exp(yhat) / tmp[:, None] - y
        else:  # L2 loss
            f = 0.5 * np.sum((yhat - y) ** 2)
            grad = yhat - y  # gradient for L2 loss

        grad_W = grad.T @ activations[-2]
        grad_b = np.sum(grad, axis=0)

        g = [(grad_W, grad_b)]

        for i in range(len(self.layer_sizes) - 2, 0, -1):
            W, b = weights[i]
            grad = grad @ W
            grad = grad * (activations[i] * (1 - activations[i]))  # gradient of logistic loss
            grad_W = grad.T @ activations[i - 1]
            grad_b = np.sum(grad, axis=0)

            g = [(grad_W, grad_b)] + g  # insert to start of list

        g = _flatten_weights(g)

        # add L2 regularization
        f += 0.5 * self.lammy * np.sum(weights_flat ** 2)
        g += self.lammy * weights_flat
        return f, g

    def fit(self, X, y):
        if y.ndim == 1:
            y = y[:, None]

        self.layer_sizes = [X.shape[1]] + self.hidden_layer_sizes + [y.shape[1]]
        self.classification = y.shape[1] > 1  # assume it's classification iff y has more than 1 column

        # random init
        scale = 0.01
        weights = list()
        for i in range(len(self.layer_sizes) - 1):
            W = scale * np.random.randn(self.layer_sizes[i + 1], self.layer_sizes[i])
            b = scale * np.random.randn(1, self.layer_sizes[i + 1])
            weights.append((W, b))
        weights_flat = _flatten_weights(weights)

        # utils.check_gradient(self, X, y, len(weights_flat), epsilon=1e-6)
        weights_flat_new, f = findMin.findMin(self.funObj, weights_flat, self.max_iter, X, y, verbose=True)
        self.weights = _unflatten_weights(weights_flat_new, self.layer_sizes)

    def fitSGD(self, X, y, minibatch_size=100, learning_rate=0.001, epoch=10):
        if y.ndim == 1:
            y = y[:, None]

        self.layer_sizes = [X.shape[1]] + self.hidden_layer_sizes + [y.shape[1]]
        self.classification = y.shape[1] > 1

        # random init
        scale = 0.01
        weights = list()
        for i in range(len(self.layer_sizes) - 1):
            W = scale * np.random.randn(self.layer_sizes[i + 1], self.layer_sizes[i])
            b = scale * np.random.randn(1, self.layer_sizes[i + 1])
            weights.append((W, b))

        weights_flat = _flatten_weights(weights)
        weights_flat_new, f = findMin.findMinSGD(self.funObj, weights_flat, self.max_iter,
                                                 minibatch_size, learning_rate, epoch, X, y, verbose=True)
        self.weights = _unflatten_weights(weights_flat_new, self.layer_sizes)

    def predict(self, X):
        for W, b in self.weights:
            Z = X @ W.T + b
            X = 1 / (1 + np.exp(-Z))
        if self.classification:
            return np.argmax(Z, axis=1)
        else:
            return Z
