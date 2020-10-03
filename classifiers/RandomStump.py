# -*- coding: utf-8 -*-
__author__ = 'xincz'

import numpy as np

from utils import common
from classifiers.DecisionStump import DecisionStumpInfoGain


class RandomStumpInfoGain(DecisionStumpInfoGain):

    def fit(self, X, y):
        # Randomly select k features.
        # This can be done by randomly permuting
        # the feature indices and taking the first k
        D = X.shape[1]
        k = int(np.floor(np.sqrt(D)))
        chosen_features = np.random.choice(D, k, replace=False)
        DecisionStumpInfoGain.fit(self, X, y, split_features=chosen_features)
