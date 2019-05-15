# -*- coding: utf-8 -*-

from __future__ import division

from math import sqrt

import numpy as np


def l2_dist(x, y):
    return np.sum(x*y) / sqrt(np.sum(x**2)*np.sum(y**2))
