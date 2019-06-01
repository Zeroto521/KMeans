# -*- coding: utf-8 -*-

import numpy as np


def check_array(array, copy=False):

    if not isinstance(array, np.ndarray):
        array = np.asarray(array)

    # copy or not
    array = array.copy() if copy else array

    return array
