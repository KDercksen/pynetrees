#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def probabilities(labels, classes):
    counts = dict(zip(*np.unique(labels, return_counts=True)))
    size = labels.shape[0]
    return np.fromiter((counts.get(c, 0) / size for c in classes), np.float)


def gini(probs):
    inverse = np.subtract(np.ones(probs.shape[0]), probs)
    return np.sum(np.multiply(probs, inverse))


def entropy(probs):
    # dummy logs
    logs = np.log2(np.where(probs > 0., probs, 10.**-10))
    return -np.sum(np.multiply(probs, logs))
