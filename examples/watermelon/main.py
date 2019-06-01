# -*- coding: utf-8 -*-

import numpy as np

from kmeans import KMeans

data = np.loadtxt('watermelon4.0.csv', delimiter=',', skiprows=1)

model = KMeans(n_clusters=2)
labels = model.fit(data)
print(labels)
