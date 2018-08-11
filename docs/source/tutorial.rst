Tutorial
==========

Basic of PQ
------------

This tutorial shows the basic usage of Nano Product Quantization (nanopq).
Product quantization (PQ) is one of the most widely used algorithms
for memory-efficient approximated nearest neighbor search,
especially in the field of computer vision.
This package contains a vanilla implementation of PQ and its improved version, Optimized Product Quantization (OPQ).

Let us first prepare 10,000 12-dim vectors for database, 2,000 vectors for training,
and a query vector. They must be np.ndarray with np.float32.

.. code-block:: python

    import nanopq
    import numpy as np

    X = np.random.random((10000, 12)).astype(np.float32)
    Xt = np.random.random((2000, 12)).astype(np.float32)
    query = np.random.random((12, )).astype(np.float32)

The basic idea of PQ is to split an input `D`-dim vector into `M` `D/M`-dim sub-vectors.
Each sub-vector is then quantized into an identifier of the nearest codeword.

First of all, a PQ class is instantiated with the number of sub-vector (`M`)
and the number of codeword for each sub-space (`Ks`).

.. code-block:: python

    pq = nanopq.PQ(M=4, Ks=256, verbose=True)

Note that `M` is a parameter to control the trade off of accuracy and memory-cost.
If you set larger `M`, you can achieve better quantization (i.e., less reconstruction error)
with more memory usage.
`Ks` specifies the number of codewords for quantization.
This is tyically 256 so that each sub-space is represented by 8 bits = 1 byte = np.uint8.
The memory cost for each pq-code is `M * log_2 Ks` bits.

Next, you need to train this quantizer by running k-means clustering for each sub-space
of the training vectors.

.. code-block:: python

    pq.fit(vecs=Xt, iter=20, seed=123)

If you do not have training data, you can simply use the database vectors
(or a subset of them) for training: ``pq.fit(vecs=X[:1000])``. After that, you can check codewords by `pq.codewords`.
Note that, alternatively, you can instantiate and train an instance in one line if you want:

.. code-block:: python

    pq = nanopq.PQ(M=4, Ks=256).fit(vecs=Xt, iter=20, seed=123)


Given this quantizer, database vectors can be encoded to PQ-codes.

.. code-block:: python

    X_code = pq.encode(vecs=X)

The resulting PQ-code (a list of identifiers) can be regarded as a memory-efficient representation of the original vector,
where the shape of `X_code` is (N, M).

For the querying phase, the asymmetric distance between the query
and the database PQ-codes can be computed efficiently.

.. code-block:: python

    dt = pq.dtable(query=query)  # dt.dtable.shape = (4, 256)
    dists = dt.adist(codes=X_code)  # (10000,)

For each query, a distance table (`dt`) is first computed online.
`dt` is an instance of `DistanceTable` class, which is a wrapper of the actual table (np.array), `dtable`.
The elements of `dt.dtable` are computed by comparing each sub-vector of the query
to the codewords for each sub-subspace.
More specifically, `dt.dtable[m][ks]` contains the squared Euclidean distance between
(1) the `m`-th sub-vector of the query and (2) the `ks`-th codeword
for the `m`-th sub-space (`pq.codewords[m][ks]`).

Given `dtable`, the asymmetric distance to each PQ-code can be efficiently computed (`adist`).
This can be achieved by simply fetching pre-computed distance value (the element of `dtable`)
using PQ-codes.

Note that the above two lines can be chained in a single line.

.. code-block:: python

    dists = pq.dtable(query=query).adist(codes=X_code)  # (10000,)


The nearest feature is the one with the minimum distance.

.. code-block:: python

    min_n = np.argmin(dists)


Note that the search result is similar to that
by the exact squared Euclidean distance.

.. code-block:: python

    # The first 30 results by PQ
    print(dists[:30])

    # The first 30 results by the exact scan
    dists_exact = np.linalg.norm(X - query, axis=1) ** 2
    print(dists_exact[:30])


Decode (reconstruction)
-------------------------------

Given PQ-codes, the original `D`-dim vectors can be
approximately reconstructed by fetching codewords

.. code-block:: python

    X_reconstructed = pq.decode(codes=X_code)  # (10000, 12)
    # The following two results should be similar
    print(X[:3])
    print(X_reconstructed[:3])



I/O by pickling
------------------

A PQ instance can be pickled. Note that PQ-codes can be pickled as well because they are
just a numpy array.

.. code-block:: python

    import pickle

    with open('pq.pkl', 'wb') as f:
        pickle.dump(pq, f)

    with open('pq.pkl', 'rb') as f:
        pq_dumped = pickle.load(f)  # pq_dumped is identical to pq



Optimized PQ (OPQ)
-------------------

Optimized Product Quantizaion (OPQ), which is an improved version of PQ, is also available
with the same interface as follows.

.. code-block:: python

    opq = nanopq.OPQ(M=4).fit(vecs=Xt, pq_iter=20, rotation_iter=10, seed=123)
    X_code = opq.encode(vecs=X)
    dists = opq.dtable(query=query).adist(codes=X_code)

The resultant codes approximate the original vectors finer,
that usually leads to the better search accuracy.
The training of OPQ will take much longer time compared to that of PQ.


Difference from PQ in faiss
------------------------------------------

Note that
`PQ is implemented in Faiss <https://github.com/facebookresearch/faiss/wiki/Faiss-building-blocks:-clustering,-PCA,-quantization#pq-encoding--decoding>`_,
whereas Faiss is one of the most powerful ANN libraries developed by the original authors of PQ.
Since Faiss is highly optimized, you should use PQ in Faiss if the runtime is your most important criteria.
The difference between PQ in `nanopq` and that in Faiss is highlighted as follows:

- Our `nanopq` can be installed simply by pip without any third party dependencies such as Intel MKL
- The core part of `nanopq` is a vanilla implementation of PQ written in a single python file.
  It would be easier to extend that for further applications.
- A standalone OPQ is implemented.
- The result of :func:`nanopq.DistanceTable.adist` is **not** sorted. This would be useful when you would like to
  know not only the nearest but also the other results.
