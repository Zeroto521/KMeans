# -*- coding: utf-8 -*-

from sklearn.datasets import load_iris

from kmeans import SeedKMeans

iris = load_iris()
data = iris.data
labels = iris.target

X = data[10:]
S = data[:10]
S_labels = labels[:10]

model = SeedKMeans(n_clusters=3)
labels = model.fit(X, S, S_labels)
print(labels)
