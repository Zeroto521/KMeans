# -*- coding: utf-8 -*-

import numpy as np

from .utils.validation import check_array

from .k_means import KMeans


class SeedKMeans(KMeans):

    def _create_center(self, X, X_train, y_train):
        sr = set(y_train)
        if len(sr) < self.n_clusters:
            data_select = X[: self.n_clusters - len(sr)]
            d = np.vstack([data_select, X_train])
            label_rest = set(range(self.n_clusters)) - sr
            l = np.hstack([list(label_rest), y_train])
            centers = self._gen_center(d, l)
        elif len(sr) == self.n_clusters:
            centers = X_train
        else:
            centers = self._gen_center(X_train, y_train)

        return centers

    def fit(self, X, X_train, y_train):
        X = check_array(X, copy=self.copy_x)
        self.cluster_centers_ = self._create_center(X, X_train, y_train)

        d = np.vstack([X_train, X])
        for _ in range(self.max_iter):
            self.labels_ = np.apply_along_axis(self._choice_center, 1, X)
            l = np.hstack([y_train, self.labels_])
            centers_new = self._gen_center(d, l)

            if np.all(abs(self.cluster_centers_ - centers_new) <= self.tol):
                break

            self.cluster_centers_ = centers_new
            self.n_iter_ += 1

        return self.labels_
