#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pynetrees.impurity as imp


def test_probabilities1():
    classes = np.array([1., 2.])
    labels = np.array([1., 2.])
    probs = imp.probabilities(labels, classes)
    expected = np.array([.5, .5])
    assert (probs == expected).all()


def test_probabilities2():
    classes = np.array([0., 1., 2.])
    labels = np.array([1., 2.])
    probs = imp.probabilities(labels, classes)
    expected = np.array([0., .5, .5])
    assert (probs == expected).all()


def test_probabilities3():
    classes = np.array([0., 1., 2.])
    labels = np.array([4., 5.])
    probs = imp.probabilities(labels, classes)
    expected = np.array([0., 0., 0.])
    assert (probs == expected).all()


def test_gini1():
    classes = np.array([1., 2.])
    labels = np.array([1., 1., 1.])
    gini = imp.gini(imp.probabilities(labels, classes))
    expected = 0.
    assert gini == expected


def test_gini2():
    classes = np.array([1., 2.])
    labels = np.array([1., 1., 2., 2.])
    gini = imp.gini(imp.probabilities(labels, classes))
    expected = .5
    assert gini == expected


def test_gini3():
    classes = np.array([1., 2., 3.])
    labels = np.array([1., 1., 2., 2., 3.])
    gini = imp.gini(imp.probabilities(labels, classes))
    expected = .64
    assert gini == expected


def test_entropy1():
    classes = np.array([1., 2.])
    labels = np.array([1., 1.])
    entropy = imp.entropy(imp.probabilities(labels, classes))
    expected = 0.
    assert entropy == expected


def test_entropy2():
    classes = np.array([1., 2.])
    labels = np.array([1., 2.])
    entropy = imp.entropy(imp.probabilities(labels, classes))
    expected = 1.
    assert entropy == expected
