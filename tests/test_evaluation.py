#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pynetrees.evaluation as ev


def test_train_test_split1():
    data = np.arange(0, 100).reshape((10, 10))
    train, test = ev.train_test_split(data, test_size=.3)
    assert train.shape[0] == 3 and test.shape[0] == 7


def test_accuracy1():
    labels = np.array([0, 0, 1, 1]).reshape((4, 1))
    preds = np.array([0, 1, 1, 1]).reshape((4, 1))
    assert ev.accuracy(preds, labels) == .75
