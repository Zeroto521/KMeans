# -*- coding: utf-8 -*-

"""
SeedKMeans
=====
It is an algorithm for choosing the initial values (or "seeds") for the k-means clustering algorithm.

Example
----------------------------
    >>> from kmeans import SeedKMeans
    >>> model = SeedKMeans()  # build model
    # Guess you have `data` which the shape is `(n, m)`. `n` is sample numbers, `m` is feature numbers.
    # The shape of `S` is similar to `data` but shape is `(n', m)`, `n'` is sample numbers.
    # `S_labels` is a vector, shape like a column or a row but the size of `S_labels` is equal to `m`
    >>> labels = model.fit(X, S, S_labels)
    >>> labels

Copyright Zeroto521
----------------------------
"""

import numpy as np

from .k_means import KMeans
from .utils.validation import check_array


class SeedKMeans(KMeans):

    def _create_center(self, X, S, S_labels):
        sr = set(S_labels)
        if len(sr) < self.n_clusters:
            xr = super(SeedKMeans, self)._create_center(X, (self.n_clusters - len(sr)))
            x = np.vstack([xr, S])
            yr = set(range(self.n_clusters)) - sr
            y = np.hstack([list(yr), S_labels])
            centers = self._gen_center(x, y)

        elif len(sr) == self.n_clusters:
            centers = S
        else:
            centers = self._gen_center(S, S_labels)

        return centers

    def fit(self, X, S, S_labels):
        X = check_array(X, copy=self.copy_x)
        S = check_array(S, copy=self.copy_x)
        x = np.vstack([S, X])
        self.cluster_centers_ = self._create_center(X, S, S_labels)

        for _ in range(self.max_iter):
            self.labels_ = np.apply_along_axis(self._choice_center, 1, X)
            y = np.hstack([S_labels, self.labels_])
            centers_new = self._gen_center(x, y)

            if np.all(abs(self.cluster_centers_ - centers_new) <= self.tol):
                break

            self.cluster_centers_ = centers_new
            self.n_iter_ += 1

        return self.labels_
