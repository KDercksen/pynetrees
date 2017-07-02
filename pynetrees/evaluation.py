#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def train_test_split(samples, test_size=0.3, seed=42):
    np.random.seed(seed)
    np.random.shuffle(samples)

    split = int(samples.shape[0] * test_size)
    return samples[:split], samples[split:]


def accuracy(predictions, samples):
    return np.sum(predictions[:, 0] == samples[:, 0]) / predictions.shape[0]
