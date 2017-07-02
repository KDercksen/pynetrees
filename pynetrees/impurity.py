#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def probabilities(labels, classes):
    size = labels.shape[0]
    if size == 0:
        return np.zeros(classes.shape[0])
    counts = dict(zip(*np.unique(labels, return_counts=True)))
    return np.fromiter((counts.get(c, 0) / size for c in classes), np.float)


def gini(probs):
    inverse = np.subtract(np.ones(probs.shape[0]), probs)
    return np.sum(np.multiply(probs, inverse))


def entropy(probs):
    # dummy logs
    logs = np.log2(np.where(probs > 0., probs, 10.**-10))
    return -np.sum(np.multiply(probs, logs))
