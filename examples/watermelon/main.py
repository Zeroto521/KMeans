# -*- coding: utf-8 -*-

import os

import numpy as np

from kmeans import KMeans

base_dir = os.path.abspath(os.path.dirname(__file__))
filepath = os.path.join(base_dir, 'watermelon4.0.csv')

data = np.loadtxt(filepath, delimiter=',', skiprows=1)

model = KMeans(n_clusters=2)
labels = model.fit(data)
print(labels)
