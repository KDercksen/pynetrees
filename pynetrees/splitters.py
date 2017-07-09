#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


class Splitter:

    def __init__(self, feature_index):
        self.feature_index = feature_index

    def feature_index(self):
        return self.feature_index

    def split(self, samples):
        # Returns (left, right) data slices depending on split criteria
        # samples should be 2D array
        raise NotImplementedError

    def split_sample_left(self, sample):
        # For a single sample, return True if the sample should go left and
        # False if it should go right.
        # sample should be 1D array
        raise NotImplementedError


class ContinuousSplitter(Splitter):

    def __init__(self, feature_index, threshold):
        super().__init__(feature_index)
        self.threshold = threshold

    def split(self, samples):
        # Go left if sample[feature index] <= threshold
        left = samples[:, self.feature_index] <= self.threshold
        return samples[left], samples[np.invert(left)]

    def split_sample_left(self, sample):
        return bool(sample[self.feature_index] <= self.threshold)

    def __str__(self):
        return 'feature {} <= {}'.format(self.feature_index, self.threshold)


class CategoricalSplitter(Splitter):

    def __init__(self, feature_index, categories):
        super().__init__(feature_index)
        self.categories = categories

    def split(self, samples):
        # Go left if sample[feature index] is in categories list
        def f(r): return r[self.feature_index] in self.categories
        left = np.apply_along_axis(f, 1, samples)
        return samples[left], samples[np.invert(left)]

    def split_sample_left(self, sample):
        return bool(sample[self.feature_index] in self.categories)

    def __str__(self):
        return 'feature {} in {}'.format(self.feature_index, self.categories)
