Tutorial
==========

Basics of PQ
------------

This tutorial demonstrates the basic usage of the Nano Product Quantization Library (nanopq).
Product quantization (PQ) is one of the most widely used algorithms
for memory-efficient approximate nearest neighbor search,
especially in the field of computer vision.
This package provides a standard implementation of PQ and its improved version, Optimized Product Quantization (OPQ).

First, let us prepare 10,000 12-dimensional vectors for the database, 2,000 vectors for training,
and a query vector. All of them must be np.ndarray with dtype np.float32.

.. code-block:: python

    import nanopq
    import numpy as np

    X = np.random.random((10000, 12)).astype(np.float32)
    Xt = np.random.random((2000, 12)).astype(np.float32)
    query = np.random.random((12, )).astype(np.float32)

The basic idea of PQ is to split an input `D`-dimensional vector into `M` sub-vectors of size `D/M`.
Each sub-vector is then quantized into the index of the nearest codeword.

First, instantiate the PQ class (:class:`nanopq.PQ`) with the number of sub-vectors (`M`)
and the number of codewords for each subspace (`Ks`).

.. code-block:: python

    pq = nanopq.PQ(M=4, Ks=256, verbose=True)

Note that `M` is a parameter that controls the trade-off between accuracy and memory cost.
If you set a larger `M`, you can achieve better quantization (i.e., lower reconstruction error)
at the expense of higher memory usage.
`Ks` specifies the number of codewords for quantization.
This is typically 256 so that each subspace is represented by 8 bits = 1 byte = np.uint8.
The memory cost for each PQ code is `M * log_2 Ks` bits.

Next, you need to train this quantizer by running k-means clustering for each subspace
using the training vectors.

.. code-block:: python

    pq.fit(vecs=Xt, iter=20, seed=123)

If you do not have training data, you can simply use the database vectors
(or a subset of them) for training: ``pq.fit(vecs=X[:1000])``. After that, you can view the codewords with `pq.codewords`.

Alternatively, you can instantiate and train an instance in one line if you prefer:

.. code-block:: python

    pq = nanopq.PQ(M=4, Ks=256).fit(vecs=Xt, iter=20, seed=123)


Given this quantizer, database vectors can be encoded into PQ codes.

.. code-block:: python

    X_code = pq.encode(vecs=X)

The resulting PQ codes (a list of indices) can be regarded as a memory-efficient representation of the original vectors,
where the shape of `X_code` is (N, M).

For the querying phase, the asymmetric distance between the query
and the database PQ codes can be computed efficiently.

.. code-block:: python

    dt = pq.dtable(query=query)  # dt.dtable.shape = (4, 256)
    dists = dt.adist(codes=X_code)  # (10000,)

For each query, a distance table (`dt`) is first computed on the fly.
`dt` is an instance of the :class:`nanopq.DistanceTable` class, which is a wrapper for the actual table (np.array), `dtable`.
The elements of `dt.dtable` are computed by comparing each sub-vector of the query
to the codewords for each subspace.
More specifically, `dt.dtable[m][ks]` contains the squared Euclidean distance between
(1) the `m`-th sub-vector of the query and (2) the `ks`-th codeword
for the `m`-th subspace (`pq.codewords[m][ks]`).

Given `dtable`, the asymmetric distance to each PQ code can be efficiently computed (`adist`).
This can be achieved by simply fetching the pre-computed distance values (the elements of `dtable`)
using the PQ codes.

Note that the above two lines can be combined into a single line.

.. code-block:: python

    dists = pq.dtable(query=query).adist(codes=X_code)  # (10000,)


The nearest feature is the one with the minimum distance.

.. code-block:: python

    min_n = np.argmin(dists)


Note that the search result is similar to that
obtained by the exact squared Euclidean distance.

.. code-block:: python

    # The first 30 results by PQ
    print(dists[:30])

    # The first 30 results by the exact scan
    dists_exact = np.linalg.norm(X - query, axis=1) ** 2
    print(dists_exact[:30])


Decode (Reconstruction)
-------------------------------

Given PQ codes, the original `D`-dimensional vectors can be
approximately reconstructed by fetching the codewords

.. code-block:: python

    X_reconstructed = pq.decode(codes=X_code)  # (10000, 12)
    # The following two results should be similar
    print(X[:3])
    print(X_reconstructed[:3])


I/O by Pickling
------------------

A PQ instance can be pickled. Note that PQ codes can be pickled as well because they are
just numpy arrays.

.. code-block:: python

    import pickle

    with open('pq.pkl', 'wb') as f:
        pickle.dump(pq, f)

    with open('pq.pkl', 'rb') as f:
        pq_dumped = pickle.load(f)  # pq_dumped is identical to pq



Optimized PQ (OPQ)
-------------------

Optimized Product Quantization (OPQ; :class:`nanopq.OPQ`), which is an improved version of PQ, is also available
with the same interface as follows.

.. code-block:: python

    opq = nanopq.OPQ(M=4).fit(vecs=Xt, pq_iter=20, rotation_iter=10, seed=123)
    X_code = opq.encode(vecs=X)
    dists = opq.dtable(query=query).adist(codes=X_code)

The resulting codes approximate the original vectors more accurately,
which usually leads to better search accuracy.
Training OPQ will take much longer compared to PQ.


Relation to PQ in Faiss
-----------------------

Note that
`PQ is implemented in Faiss <https://github.com/facebookresearch/faiss/wiki/Faiss-building-blocks:-clustering,-PCA,-quantization#pq-encoding--decoding>`_,
which is one of the most powerful ANN libraries developed by the original authors of PQ:

- `faiss.ProductQuantizer <https://github.com/facebookresearch/faiss/blob/master/ProductQuantizer.h>`_: The core component of PQ.
- `faiss.IndexPQ <https://github.com/facebookresearch/faiss/blob/master/IndexPQ.h>`_: The search interface. IndexPQ = ProductQuantizer + PQ-codes.

Since Faiss is highly optimized, you should use PQ in Faiss if runtime is your most important criterion.
The differences between PQ in `nanopq` and that in Faiss are highlighted as follows:

- Our `nanopq` can be installed simply via pip without any third-party dependencies such as Intel MKL.
- The core part of `nanopq` is a standard implementation of PQ written in a single Python file.
  It is easier to extend for further applications.
- A standalone OPQ is implemented.
- The result of :func:`nanopq.DistanceTable.adist` is **not** sorted. This is useful when you want to
  know not only the nearest but also other results.
- The accuracy (reconstruction error) of `nanopq.PQ` and that of `faiss.IndexPQ` are `almost the same <https://github.com/matsui528/nanopq/blob/master/tests/test_convert_faiss.py>`_.

You can convert an instance of `nanopq.PQ` to/from that of `faiss.IndexPQ`
using :func:`nanopq.nanopq_to_faiss` or :func:`nanopq.faiss_to_nanopq`.

.. code-block:: python

    # nanopq -> faiss
    pq_nanopq = nanopq.PQ(M).fit(vecs=Xt)
    pq_faiss = nanopq.nanopq_to_faiss(pq_nanopq)  # faiss.IndexPQ

    # faiss -> nanopq
    import faiss
    pq_faiss2 = faiss.IndexPQ(D, M, nbits)
    pq_faiss2.train(x=Xt)
    pq_faiss2.add(x=Xb)
    # pq_nanopq2 is an instance of nanopq.PQ.
    # Cb is encoded vectors
    pq_nanopq2, Cb = nanopq.faiss_to_nanopq(pq_faiss2)
