# -*- coding: utf-8 -*-

import numpy as np

from .utils.distance import l2_dist
from .utils.validation import check_array


class KMeans(object):

    def __init__(self, n_clusters=3, max_iter=1000, tol=1e-4,
                 compute_distances=l2_dist, random_state=None, copy_x=True):

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.compute_distances = compute_distances
        self.tol = tol
        self.random_state = random_state
        self.copy_x = copy_x
        self.n_iter_ = 0

    def _create_center(self, X):
        inds = np.random.choice(len(X), self.n_clusters)
        centers = X[inds]
        return centers

    def _choice_center(self, row):
        dis = [self.compute_distances(row, center)
               for center in self.cluster_centers_]
        labels = np.argmax(dis)

        return labels

    def _gen_center(self, X, labels):
        centers = np.zeros((self.n_clusters, len(X[0])))
        for i in range(self.n_clusters):
            centers[i] = np.mean(X[labels == i], axis=0)

        return centers

    def fit(self, X):
        X = check_array(X, copy=self.copy_x)
        self.cluster_centers_ = self._create_center(X)

        for _ in range(self.max_iter):
            self.labels_ = np.apply_along_axis(self._choice_center, 1, X)
            centers_new = self._gen_center(X, self.labels_)

            if np.all(abs(self.cluster_centers_ - centers_new) <= self.tol):
                break

            self.cluster_centers_ = centers_new
            self.n_iter_ += 1

        return self.labels_

    def predict(self, X):
        X = check_array(X, copy=self.copy_x)
        return np.apply_along_axis(self._choice_center, 1, X)
