import numpy as np
from scipy.cluster.vq import kmeans2, vq
from typing import Literal, Optional


def dist_l2(q, x):
    return np.linalg.norm(q - x, ord=2, axis=1) ** 2


def dist_ip(q, x):
    return q @ x.T


metric_function_map = {"l2": dist_l2, "dot": dist_ip}


class PQ(object):
    """Pure python implementation of Product Quantization (PQ) [Jegou11]_.

    For the indexing phase of database vectors,
    a `D`-dim input vector is divided into `M` `D`/`M`-dim sub-vectors.
    Each sub-vector is quantized into a small integer via `Ks` codewords.
    For the querying phase, given a new `D`-dim query vector, the distance beween the query
    and the database PQ-codes are efficiently approximated via Asymmetric Distance.

    All vectors must be np.ndarray with np.float32

    .. [Jegou11] H. Jegou et al., "Product Quantization for Nearest Neighbor Search", IEEE TPAMI 2011

    Args:
        M (int): The number of sub-space
        Ks (int): The number of codewords for each subspace
            (typically 256, so that each sub-vector is quantized
            into 8 bits = 1 byte = uint8)
        metric (Literal["l2", "dot"]): Type of metric used among vectors (either 'l2' or 'dot')
            Note that even for 'dot', kmeans and encoding are performed in the Euclidean space.
        verbose (bool): Verbose flag

    Attributes:
        M (int): The number of sub-space
        Ks (int): The number of codewords for each subspace
        metric (Literal["l2", "dot"]): Type of metric used among vectors
        verbose (bool): Verbose flag
        code_dtype (object): dtype of PQ-code. Either np.uint{8, 16, 32}
        codewords (np.ndarray): shape=(M, Ks, Ds) with dtype=np.float32.
            codewords[m][ks] means ks-th codeword (Ds-dim) for m-th subspace
        Ds (int): The dim of each sub-vector, i.e., Ds=D/M

    """

    def __init__(self, M, Ks=256, metric: Literal["l2", "dot"] = "l2", verbose=True):
        assert 0 < Ks <= 2**32
        assert metric in ["l2", "dot"]
        self.M, self.Ks, self.metric, self.verbose = M, Ks, metric, verbose
        self.code_dtype = (
            np.uint8 if Ks <= 2**8 else (np.uint16 if Ks <= 2**16 else np.uint32)
        )
        self.codewords: Optional[np.ndarray] = None
        self.Ds: Optional[int] = None

        if verbose:
            print(
                "M: {}, Ks: {}, metric : {}, code_dtype: {}".format(
                    M, Ks, self.code_dtype, metric
                )
            )

    def __eq__(self, other):
        if isinstance(other, PQ):
            return (
                self.M,
                self.Ks,
                self.metric,
                self.verbose,
                self.code_dtype,
                self.Ds,
            ) == (
                other.M,
                other.Ks,
                other.metric,
                other.verbose,
                other.code_dtype,
                other.Ds,
            ) and np.array_equal(self.codewords, other.codewords)
        else:
            return False

    def fit(
        self,
        vecs,
        iter=20,
        seed=123,
        minit: Literal["random", "++", "points", "matrix"] = "points",
    ):
        """Given training vectors, run k-means for each sub-space and create
        codewords for each sub-space.

        This function should be run once first of all.

        Args:
            vecs (np.ndarray): Training vectors with shape=(N, D) and dtype=np.float32.
            iter (int): The number of iteration for k-means
            seed (int): The seed for random process
            minit (Literal["random", "++", "points", "matrix"]): The method for initialization of centroids for k-means (either 'random', '++', 'points', 'matrix')

        Returns:
            object: self

        """
        assert vecs.dtype == np.float32
        assert vecs.ndim == 2
        N, D = vecs.shape
        assert self.Ks < N, "the number of training vector should be more than Ks"
        assert D % self.M == 0, "input dimension must be dividable by M"
        assert minit in ["random", "++", "points", "matrix"]
        self.Ds = int(D / self.M)

        np.random.seed(seed)
        if self.verbose:
            print("iter: {}, seed: {}".format(iter, seed))

        # [m][ks][ds]: m-th subspace, ks-the codeword, ds-th dim
        self.codewords = np.zeros((self.M, self.Ks, self.Ds), dtype=np.float32)
        for m in range(self.M):
            if self.verbose:
                print("Training the subspace: {} / {}".format(m, self.M))
            vecs_sub = vecs[:, m * self.Ds : (m + 1) * self.Ds]
            self.codewords[m], _ = kmeans2(vecs_sub, self.Ks, iter=iter, minit=minit)
        return self

    def encode(self, vecs):
        """Encode input vectors into PQ-codes.

        Args:
            vecs (np.ndarray): Input vectors with shape=(N, D) and dtype=np.float32.

        Returns:
            np.ndarray: PQ codes with shape=(N, M) and dtype=self.code_dtype

        """
        assert vecs.dtype == np.float32
        assert vecs.ndim == 2
        N, D = vecs.shape
        assert D == self.Ds * self.M, "input dimension must be Ds * M"
        assert self.Ds is not None, "Ds must be set by fit() before encode()"
        assert isinstance(self.Ds, int), "Ds must be an integer"
        assert self.codewords is not None, "codewords must be set by fit() before encode()"

        # codes[n][m] : code of n-th vec, m-th subspace
        codes = np.empty((N, self.M), dtype=self.code_dtype)
        for m in range(self.M):
            if self.verbose:
                print("Encoding the subspace: {} / {}".format(m, self.M))
            vecs_sub = vecs[:, m * self.Ds : (m + 1) * self.Ds]
            codes[:, m], _ = vq(vecs_sub, self.codewords[m])

        return codes

    def decode(self, codes):
        """Given PQ-codes, reconstruct original D-dimensional vectors
        approximately by fetching the codewords.

        Args:
            codes (np.ndarray): PQ-cdoes with shape=(N, M) and dtype=self.code_dtype.
                Each row is a PQ-code

        Returns:
            np.ndarray: Reconstructed vectors with shape=(N, D) and dtype=np.float32

        """
        assert codes.ndim == 2
        N, M = codes.shape
        assert M == self.M
        assert codes.dtype == self.code_dtype
        assert self.Ds is not None, "Ds must be set by fit() before decode()"
        assert self.codewords is not None, "codewords must be set by fit() before decode()"

        vecs = np.empty((N, self.Ds * self.M), dtype=np.float32)
        for m in range(self.M):
            vecs[:, m * self.Ds : (m + 1) * self.Ds] = self.codewords[m][codes[:, m], :]

        return vecs

    def dtable(self, query):
        """Compute a distance table for a query vector.
        The distances are computed by comparing each sub-vector of the query
        to the codewords for each sub-subspace.
        `dtable[m][ks]` contains the squared Euclidean distance between
        the `m`-th sub-vector of the query and the `ks`-th codeword
        for the `m`-th sub-space (`self.codewords[m][ks]`).

        Args:
            query (np.ndarray): Input vector with shape=(D, ) and dtype=np.float32

        Returns:
            nanopq.DistanceTable:
                Distance table. which contains
                dtable with shape=(M, Ks) and dtype=np.float32

        """
        assert query.dtype == np.float32
        assert query.ndim == 1, "input must be a single vector"
        (D,) = query.shape
        assert D == self.Ds * self.M, "input dimension must be Ds * M"
        assert self.Ds is not None, "Ds must be set by fit() before dtable()"
        assert self.codewords is not None, "codewords must be set by fit() before dtable()"

        # dtable[m] : distance between m-th subvec and m-th codewords (m-th subspace)
        # dtable[m][ks] : distance between m-th subvec and ks-th codeword of m-th codewords
        dtable = np.empty((self.M, self.Ks), dtype=np.float32)
        for m in range(self.M):
            query_sub = query[m * self.Ds : (m + 1) * self.Ds]
            dtable[m, :] = metric_function_map[self.metric](
                query_sub, self.codewords[m]
            )

            # In case of L2, the above line would be:
            # dtable[m, :] = np.linalg.norm(self.codewords[m] - query_sub, axis=1) ** 2

        return DistanceTable(dtable, metric=self.metric)


class DistanceTable(object):
    """Distance table from query to codewords.
    Given a query vector, a PQ/OPQ instance compute this DistanceTable class
    using :func:`PQ.dtable` or :func:`OPQ.dtable`.
    The Asymmetric Distance from query to each database codes can be computed
    by :func:`DistanceTable.adist`.

    Args:
        dtable (np.ndarray): Distance table with shape=(M, Ks) and dtype=np.float32
            computed by :func:`PQ.dtable` or :func:`OPQ.dtable`
        metric (Literal["l2", "dot"]): metric type to calculate distance

    Attributes:
        dtable (np.ndarray): Distance table with shape=(M, Ks) and dtype=np.float32.
            Note that dtable[m][ks] contains the squared Euclidean distance between
            (1) m-th sub-vector of query and (2) ks-th codeword for m-th subspace.

    """

    def __init__(self, dtable, metric: Literal["l2", "dot"] = "l2"):
        assert dtable.ndim == 2
        assert dtable.dtype == np.float32
        assert metric in ["l2", "dot"]
        self.dtable = dtable
        self.metric = metric

    def adist(self, codes):
        """Given PQ-codes, compute Asymmetric Distances between the query (self.dtable)
        and the PQ-codes.

        Args:
            codes (np.ndarray): PQ codes with shape=(N, M) and
                dtype=pq.code_dtype where pq is a pq instance that creates the codes

        Returns:
            np.ndarray: Asymmetric Distances with shape=(N, ) and dtype=np.float32

        """

        assert codes.ndim == 2
        N, M = codes.shape
        assert M == self.dtable.shape[0]

        # Fetch distance values using codes.
        dists = np.sum(self.dtable[range(M), codes], axis=1)

        # The above line is equivalent to the followings:
        # dists = np.zeros((N, )).astype(np.float32)
        # for n in range(N):
        #     for m in range(M):
        #         dists[n] += self.dtable[m][codes[n][m]]

        return dists
