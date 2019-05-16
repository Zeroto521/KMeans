# -*- coding: utf-8 -*-

import numpy as np

from .k_means import KMeans
from .utils.validation import check_array


class SeedKMeans(KMeans):

    def _create_center(self, X, S, S_labels):
        sr = set(S_labels)
        if len(sr) < self.n_clusters:
            xr = super()._create_center(X, (self.n_clusters - len(sr)))
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
