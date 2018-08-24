from .pq import PQ, DistanceTable
import numpy as np


class OPQ(object):
    """Pure python implementation of Optimized Product Quantization (OPQ) [Ge14]_.

    OPQ is a simple extension of PQ.
    The best rotation matrix `R` is prepared using training vectors.
    Each input vector is rotated via `R`, then quantized into PQ-codes
    in the same manner as the original PQ.

    .. [Ge14] T. Ge et al., "Optimized Product Quantization", IEEE TPAMI 2014

    Args:
        M (int): The number of sub-spaces
        Ks (int): The number of codewords for each subspace (typically 256, so that each sub-vector is quantized
            into 256 bits = 1 byte = uint8)
        verbose (bool): Verbose flag

    Attributes:
        R (np.ndarray): Rotation matrix with the shape=(D, D) and dtype=np.float32


    """
    def __init__(self, M, Ks=256, verbose=True):
        self.pq = PQ(M, Ks, verbose)
        self.R = None

    def __eq__(self, other):
        if isinstance(other, OPQ):
            return self.pq == other.pq and np.array_equal(self.codewords, other.codewords)
        else:
            return False

    @property
    def M(self):
        """int: The number of sub-space"""
        return self.pq.M

    @property
    def Ks(self):
        """int: The number of codewords for each subspace"""
        return self.pq.Ks

    @property
    def verbose(self):
        """bool: Verbose flag"""
        return self.pq.verbose

    @property
    def code_dtype(self):
        """object: dtype of PQ-code. Either np.uint{8, 16, 32}"""
        return self.pq.code_dtype

    @property
    def codewords(self):
        """np.ndarray: shape=(M, Ks, Ds) with dtype=np.float32.
        codewords[m][ks] means ks-th codeword (Ds-dim) for m-th subspace
        """
        return self.pq.codewords

    @property
    def Ds(self):
        """int: The dim of each sub-vector, i.e., Ds=D/M"""
        return self.pq.Ds

    def fit(self, vecs, pq_iter=20, rotation_iter=10, seed=123):
        """Given training vectors, this function alternatively trains
        (a) codewords and (b) a rotation matrix.
        The procedure of training codewords is same as :func:`PQ.fit`.
        The rotation matrix is computed so as to minimize the quantization error
        given codewords (Orthogonal Procrustes problem)

        This function is a translation from the original MATLAB implementation to that of python
        http://kaiminghe.com/cvpr13/index.html

        If you find the error message is messy, please turn off the verbose flag, then
        you can see the reduction of error for each iteration clearly

        Args:
            vecs: (np.ndarray): Training vectors with shape=(N, D) and dtype=np.float32.
            pq_iter (int): The number of iteration for k-means
            rotation_iter (int): The number of iteration for leraning rotation
            seed (int): The seed for random process

        Returns:
            object: self

        """
        assert vecs.dtype == np.float32
        assert vecs.ndim == 2
        _, D = vecs.shape
        self.R = np.eye(D, dtype=np.float32)

        for i in range(rotation_iter):
            print("OPQ rotation training: {} / {}".format(i, rotation_iter))
            X = vecs @ self.R

            # (a) Train codewords
            pq_tmp = PQ(M=self.M, Ks=self.Ks, verbose=self.verbose)
            if i == rotation_iter - 1:
                # In the final loop, run the full training
                pq_tmp.fit(X, iter=pq_iter, seed=seed)
            else:
                # During the training for OPQ, just run one-pass (iter=1) PQ training
                pq_tmp.fit(X, iter=1, seed=seed)

            # (b) Update a rotation matrix R
            X_ = pq_tmp.decode(pq_tmp.encode(X))
            U, s, V = np.linalg.svd(vecs.T @ X_)
            print("==== Reconstruction error:", np.linalg.norm(X - X_, 'fro'), "====")
            if i == rotation_iter - 1:
                self.pq = pq_tmp
                break
            else:
                self.R = U @ V

        return self

    def rotate(self, vecs):
        """Rotate input vector(s) by the rotation matrix.`

        Args:
            vecs (np.ndarray): Input vector(s) with dtype=np.float32.
                The shape can be a single vector (D, ) or several vectors (N, D)

        Returns:
            np.ndarray: Rotated vectors with the same shape and dtype to the input vecs.

        """
        assert vecs.dtype == np.float32
        assert vecs.ndim in [1, 2]

        if vecs.ndim == 2:
            return vecs @ self.R
        elif vecs.ndim == 1:
            return (vecs.reshape(1, -1) @ self.R).reshape(-1)

    def encode(self, vecs):
        """Rotate input vectors by :func:`OPQ.rotate`, then encode them via :func:`PQ.encode`.

        Args:
            vecs (np.ndarray): Input vectors with shape=(N, D) and dtype=np.float32.

        Returns:
            np.ndarray: PQ codes with shape=(N, M) and dtype=self.code_dtype

        """
        return self.pq.encode(self.rotate(vecs))

    def decode(self, codes):
        """Given PQ-codes, reconstruct original D-dimensional vectors via :func:`PQ.decode`,
        and applying an inverse-rotation.

        Args:
            codes (np.ndarray): PQ-cdoes with shape=(N, M) and dtype=self.code_dtype.
                Each row is a PQ-code

        Returns:
            np.ndarray: Reconstructed vectors with shape=(N, D) and dtype=np.float32

        """
        # Because R is a rotation matrix (R^t * R = I), R^-1 should be R^t
        return self.pq.decode(codes) @ self.R.T

    def dtable(self, query):
        """Compute a distance table for a query vector. The query is
        first rotated by :func:`OPQ.rotate`, then DistanceTable is computed by :func:`PQ.dtable`.

        Args:
            query (np.ndarray): Input vector with shape=(D, ) and dtype=np.float32

        Returns:
            nanopq.DistanceTable:
                Distance table. which contains
                dtable with shape=(M, Ks) and dtype=np.float32

        """
        return self.pq.dtable(self.rotate(query))










