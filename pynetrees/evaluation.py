#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def train_test_split(samples, test_size=0.3, seed=42):
    if samples.ndim != 2:
        raise ValueError('Only 2D data is supported')
    np.random.seed(seed)
    np.random.shuffle(samples)
    split = int(samples.shape[0] * test_size)
    return samples[:split], samples[split:]


def accuracy(predictions, samples):
    p = predictions
    if predictions.ndim == 1:
        p = p.reshape((p.shape[0], 1))
    return np.sum(p[:, 0] == samples[:, 0]) / p.shape[0]
