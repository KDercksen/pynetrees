#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pynetrees.splitters as sp


def test_categorical1():
    c = sp.CategoricalSplitter(0, np.array([1.]))
    samples = np.array([[0.], [1.]])
    left, right = c.split(samples)
    assert left.size == 1 and right.size == 1


def test_categorical2():
    c = sp.CategoricalSplitter(0, np.array([1.]))
    samples = np.array([[1.], [1.]])
    left, right = c.split(samples)
    assert left.size == 2 and right.size == 0


def test_categorical3():
    c = sp.CategoricalSplitter(0, np.array([1.]))
    samples = np.array([[0.], [0.]])
    left, right = c.split(samples)
    assert left.size == 0 and right.size == 2


def test_categorical4():
    c = sp.CategoricalSplitter(0, np.array([1.]))
    samples = np.array([0.])
    left = c.split_sample_left(samples)
    assert left is False


def test_categorical5():
    c = sp.CategoricalSplitter(0, np.array([1.]))
    samples = np.array([1.])
    left = c.split_sample_left(samples)
    assert left is True


def test_continuous1():
    c = sp.ContinuousSplitter(0, .5)
    samples = np.array([[.2], [.8]])
    left, right = c.split(samples)
    assert left.size == 1 and right.size == 1


def test_continuous2():
    c = sp.ContinuousSplitter(0, .5)
    samples = np.array([[.9], [.8]])
    left, right = c.split(samples)
    assert left.size == 0 and right.size == 2


def test_continuous3():
    c = sp.ContinuousSplitter(0, .5)
    samples = np.array([[.3], [.4]])
    left, right = c.split(samples)
    assert left.size == 2 and right.size == 0


def test_continuous4():
    c = sp.ContinuousSplitter(0, .5)
    samples = np.array([.3])
    left = c.split_sample_left(samples)
    assert left is True


def test_continuous5():
    c = sp.ContinuousSplitter(0, .5)
    samples = np.array([.8])
    left = c.split_sample_left(samples)
    assert left is False
