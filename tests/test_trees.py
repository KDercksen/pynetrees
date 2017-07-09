#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from pynetrees.trees import DecisionTree, Node
from pynetrees.splitters import ContinuousSplitter
from pytest import raises


def test_node_constructor():
    n = Node(0, .8, np.array([.2, .2, .4, .2]), None, None, None, True)
    assert n.error == .6
    assert n.prediction == 2


def test_node_predict():
    n = Node(0, .8, np.array([.2, .2, .4, .2]), None, None, None, True)
    p = n.predict(np.array([1., 1., 1., 1.]))
    assert p == 2


def test_node_predict_probs():
    probs = np.array([.2, .2, .4, .2])
    n = Node(0, .8, probs, None, None, None, True)
    p = n.predict(np.array([1., 1., 1., 1.]), prob=True)
    assert np.array_equal(probs, p)


def test_multiple_node_predict():
    splitter = ContinuousSplitter(0, 0.5)
    left = Node(1, 0., np.array([1., 0., 0., 0.]), None, None, None, True)
    n = Node(0, .8, np.array([.2, .2, .4, .2]), splitter, left, None, False)
    p = n.predict(np.array([.2, 1., 1., 1.]))
    assert p == 0


def test_multiple_node_predict_probs():
    splitter = ContinuousSplitter(0, 0.5)
    leafprobs = np.array([1., 0., 0., 0.])
    left = Node(1, 0., leafprobs, None, None, None, True)
    n = Node(0, .8, np.array([.2, .2, .4, .2]), splitter, left, None, False)
    p = n.predict(np.array([.2, 1., 1., 1.]), prob=True)
    assert np.array_equal(leafprobs, p)


def test_decisiontree_unsupported_metric():
    with raises(KeyError):
        DecisionTree(metric='fuckton')


def test_decisiontree_unfit_prediction():
    t = DecisionTree()
    with raises(ValueError):
        t.predict(np.array([1., 1.]))


def test_decisiontree_fit_ndim():
    t = DecisionTree()
    with raises(ValueError):
        t.fit(np.array([1., 1., 1., 1.]))


def test_decisiontree_max_depth_negative():
    t = DecisionTree(max_depth=-1)
    assert t.max_depth == 0


def test_decisiontree_max_depth_regular():
    t = DecisionTree(max_depth=10)
    assert t.max_depth == 10


def test_decisiontree_unsupported_strat():
    with raises(KeyError):
        DecisionTree(subset_strategy='fuckoff')
