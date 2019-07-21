# -*- coding: utf-8 -*-

"""
KMeans
=====
k-means clustering aims to partition n observations into k clusters
in which each observation belongs to the cluster with the nearest mean,
serving as a prototype of the cluster.

Example
----------------------------
    >>> from kmeans import KMeans
    >>> model = KMeans()  # build model
    >>> labels = model.fit(data)  # Guess you have `data` which the shape is `(n, m)`. `n` is sample numbers, `m` is feature numbers.
    >>> labels

Copyright Zeroto521
----------------------------
"""

import numpy as np

from .utils.distance import l2_dist
from .utils.validation import check_array


class KMeans(object):

    def __init__(self, n_clusters=3, max_iter=1000, tol=1e-4,
                 dis_func=l2_dist, random_state=None, copy_x=True):

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.dis_func = dis_func
        self.tol = tol
        self.random_state = random_state
        self.copy_x = copy_x
        self.n_iter_ = 0

        self.labels_ = None
        self.cluster_centers_ = None

    def _create_center(self, X, numbers):
        inds = np.random.choice(len(X), numbers, replace=False)
        centers = X[inds]

        return centers

    def _cal_distance(self, center, row):
        return self.dis_func(center, row)

    def _choice_center(self, row):
        dis = np.apply_along_axis(
            self._cal_distance, 1, self.cluster_centers_, row=row)
        labels = np.argmax(dis)  # FIXME: some problem are here, strange.

        return labels

    def _gen_center(self, X, labels):
        centers = np.zeros((self.n_clusters, len(X[0])))
        for i in range(self.n_clusters):
            centers[i] = np.mean(X[labels == i], axis=0)

        return centers

    def fit(self, X):
        X = check_array(X, copy=self.copy_x)
        self.cluster_centers_ = self._create_center(X, self.n_clusters)

        for self.n_iter_ in range(self.max_iter):
            self.labels_ = np.apply_along_axis(self._choice_center, 1, X)
            centers_new = self._gen_center(X, self.labels_)

            if np.all(abs(self.cluster_centers_ - centers_new) <= self.tol):
                break

            self.cluster_centers_ = centers_new

        return self.labels_

    def predict(self, X):
        X = check_array(X, copy=self.copy_x)
        return np.apply_along_axis(self._choice_center, 1, X)
