# KMeans

A Python implementation of base KMEANS, SEED-KMEANS algorithm.

[![Build Status](https://travis-ci.com/Zeroto521/KMeans.svg?branch=master)](https://travis-ci.com/Zeroto521/KMeans) [![codecov](https://codecov.io/gh/Zeroto521/kmeans/branch/master/graph/badge.svg)](https://codecov.io/gh/Zeroto521/kmeans)

## Prerequisites

-   numpy

> More details for [requirements](requirements.txt) file.

## Installation

```bash
>>> git clone https://github.com/Zeroto521/KMeans.git
>>> cd KMeans
>>> python setup.py install
```

## Examples

```bash
>>> from kmeans import KMeans
>>> model = KMeans()  # build model
>>> labels = model.fit(data)  # Guess you have `data` which the shape is `(n, m)`. `n` is sample numbers, `m` is feature numbers.
>>> labels
```

> more examples can see [here](./examples).

## License

MIT License. [@Zeroto521](https://github.com/Zeroto521)

## TODO

-   [ ] all kinds of distane
-   [ ] write doc
-   [ ] Optimized sparse matrix
-   [ ] add generalized kmeans algothrims
