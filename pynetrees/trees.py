#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from .impurity import entropy, gini, probabilities
from .splitters import ContinuousSplitter


METRICS = {
    'gini': gini,
    'entropy': entropy,
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

    def __init__(self, metric='gini'):
        self.tree = None
        self.n_features = None
        self.n_classes = None

        self.metric = METRICS.get(metric)
        if self.metric is None:
            supported = list(METRICS.keys())
            raise KeyError('Supported metrics: {}'.format(supported))

    def fit(self, samples):
        if samples.ndim != 2:
            raise ValueError('Only 2D data is supported')
        self.classes = np.sort(np.unique(samples[:, 0]))
        self.n_classes = self.classes.size
        self.n_features = samples.shape[1] - 1

        idcount = [-1]

        def _build_tree(samples):
            idcount[0] += 1
            probs = probabilities(samples[:, 0], self.classes)
            impurity = self.metric(probs)
            if np.nonzero(probs)[0].size == 1:
                return Node(idcount[0], impurity, probs, None, None, None,
                            True)
            else:
                best_splits = []
                for f in range(1, self.n_features + 1):
                    sorted_samples = samples[samples[:, f].argsort()]
                    # create splits by convolving with [.5 .5]
                    # this gives all values between the values in fvals.
                    # example [1, 2, 3] -> [1.5, 2.5]
                    weights = np.repeat(1., 2) / 2
                    splitvs = np.convolve(sorted_samples[:, f], weights)[1:-1]
                    impurities = []
                    for s in splitvs:
                        splitter = ContinuousSplitter(f, s)
                        left, right = splitter.split(samples)
                        left_probs = probabilities(left[:, 0], self.classes)
                        right_probs = probabilities(right[:, 0], self.classes)
                        v = self.metric(left_probs) + self.metric(right_probs)
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
                                _build_tree(left), _build_tree(right), False)

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
            m = 'DecisionTree <n_features: {}, n_classes: {}, metric: {}>\n' \
                .format(self.n_features, self.n_classes, self.metric.__name__)
            m += str(self.tree)
            return m
