#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .impurity import entropy, gini, probabilities
from .splitters import ContinuousSplitter
from math import log2, sqrt
from random import sample
import numpy as np


METRICS = {
    'gini': gini,
    'entropy': entropy,
}


FEATURE_SUBSET_STRATS = {
    'all': lambda x: range(1, x + 1),
    'sqrt': lambda x: sample(range(1, x + 1), int(sqrt(x))),
    'log2': lambda x: sample(range(1, x+1), int(log2(x))),
}


class Node:

    def __init__(self, ID, impurity, distribution, splitter, left_child,
                 right_child, is_leaf):
        self.ID = ID
        self.impurity = impurity
        self.distribution = distribution
        self.error = 1. - np.amax(distribution)
        self.prediction = np.argmax(distribution)
        self.splitter = splitter
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = is_leaf

    def predict(self, sample, prob=False):
        if self.is_leaf:
            return self.distribution if prob else self.prediction
        else:
            left = self.splitter.split_sample_left(sample)
            if left:
                return self.left_child.predict(sample, prob=prob)
            else:
                return self.right_child.predict(sample, prob=prob)

    def __repr__(self):
        if not self.is_leaf:
            return 'node {}: {}'.format(self.ID, self.splitter)
        else:
            return 'node {}: predict {}'.format(self.ID, self.prediction)

    def __str__(self, level=0):
        msg = '>' * level + repr(self) + '\n'
        if not self.is_leaf:
            msg += self.left_child.__str__(level+1)
            msg += self.right_child.__str__(level+1)
        return msg


class DecisionTree:

    def __init__(self, metric='gini', max_depth=5, subset_strategy='all'):
        self.tree = None
        self.n_features = None
        self.n_classes = None

        self.metric = metric
        self.metric_f = METRICS.get(metric)
        if self.metric_f is None:
            supported = list(METRICS.keys())
            raise KeyError('Supported metrics: {}'.format(supported))

        self.max_depth = max_depth if max_depth >= 0 else 0

        self.subset_strategy = subset_strategy
        self.subset_strategy_f = FEATURE_SUBSET_STRATS.get(subset_strategy)
        if self.subset_strategy_f is None:
            supported = list(FEATURE_SUBSET_STRATS.keys())
            raise KeyError('Supported subset strategies: {}'.format(supported))

    def fit(self, samples):
        if samples.ndim != 2:
            raise ValueError('Only 2D data is supported')
        self.classes = np.sort(np.unique(samples[:, 0]))
        self.n_classes = self.classes.size
        self.n_features = samples.shape[1] - 1

        idcount = [-1]

        def _build_tree(samples, depth=0):
            idcount[0] += 1
            probs = probabilities(samples[:, 0], self.classes)
            impurity = self.metric_f(probs)
            if np.nonzero(probs)[0].size == 1 or depth == self.max_depth:
                return Node(idcount[0], impurity, probs, None, None, None,
                            True)
            else:
                best_splits = []
                for f in self.subset_strategy_f(self.n_features):
                    sorted_samples = samples[samples[:, f].argsort()]
                    fsplits = sorted_samples[:, f]
                    impurities = []
                    for s in fsplits:
                        splitter = ContinuousSplitter(f, s)
                        left, right = splitter.split(samples)
                        left_p = probabilities(left[:, 0], self.classes)
                        right_p = probabilities(right[:, 0], self.classes)
                        v = self.metric_f(left_p) + self.metric_f(right_p)
                        impurities.append((s, v))
                    # best split for this feature:
                    s, val = min(impurities, key=lambda x: x[1])
                    # append to global best splits
                    best_splits.append((f, s, val))
                # global best split:
                f, s, val = min(best_splits, key=lambda x: x[2])
                splitter = ContinuousSplitter(f, s)
                left, right = splitter.split(samples)

                if left.size == 0 or right.size == 0:
                    return Node(idcount[0], impurity, probs, None, None,
                                None, True)
                else:
                    return Node(idcount[0], impurity, probs, splitter,
                                _build_tree(left, depth=depth+1),
                                _build_tree(right, depth=depth+1), False)

        self.tree = _build_tree(samples)
        return self

    def predict(self, samples, prob=False):
        if not self.tree:
            raise ValueError('DecisionTree must be fitted first')
        if samples.ndim == 1:
            return self.tree.predict(samples, prob=prob)
        else:
            p = np.vectorize(self.tree.predict)
            return p(samples, prob=prob)

    def __str__(self):
        if not self.tree:
            return 'DecisionTree was not fitted'
        else:
            m = 'DecisionTree <' \
                   'n_features: {}, ' \
                   'n_classes: {}, ' \
                   'metric: {}, ' \
                   'subset_strategy: {}' \
                   '>\n'.format(self.n_features,
                                self.n_classes,
                                self.metric,
                                self.subset_strategy,
                                )
            m += str(self.tree)
            return m
