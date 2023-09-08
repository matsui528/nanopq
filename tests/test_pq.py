import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
import unittest

import nanopq
import numpy as np


class TestSuite(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)

    def test_instantiate(self):
        pq1 = nanopq.PQ(M=4, Ks=256)
        pq2 = nanopq.PQ(M=4, Ks=500)
        pq3 = nanopq.PQ(M=4, Ks=2**16 + 10)
        self.assertEqual(pq1.code_dtype, np.uint8)
        self.assertEqual(pq2.code_dtype, np.uint16)
        self.assertEqual(pq3.code_dtype, np.uint32)

    def test_fit(self):
        N, D, M, Ks = 100, 12, 4, 10
        X = np.random.random((N, D)).astype(np.float32)
        pq = nanopq.PQ(M=M, Ks=Ks)
        pq.fit(X)
        self.assertEqual(pq.Ds, D / M)
        self.assertEqual(pq.codewords.shape, (M, Ks, D / M))

        pq2 = nanopq.PQ(M=M, Ks=Ks).fit(X)  # Can be called as a chain
        self.assertTrue(np.allclose(pq.codewords, pq2.codewords))

    def test_eq(self):
        import copy

        N, D, M, Ks = 100, 12, 4, 10
        X = np.random.random((N, D)).astype(np.float32)
        pq1 = nanopq.PQ(M=M, Ks=Ks)
        pq2 = nanopq.PQ(M=M, Ks=Ks)
        pq3 = copy.deepcopy(pq1)
        pq4 = nanopq.PQ(M=M, Ks=2 * Ks)
        self.assertTrue(pq1 == pq1)
        self.assertTrue(pq1 == pq2)
        self.assertTrue(pq1 == pq3)
        self.assertTrue(pq1 != pq4)

        pq1.fit(X)
        pq2.fit(X)
        pq3 = copy.deepcopy(pq1)
        pq4.fit(X)
        self.assertTrue(pq1 == pq1)
        self.assertTrue(pq1 == pq2)
        self.assertTrue(pq1 == pq3)
        self.assertTrue(pq1 != pq4)

    def test_encode_decode(self):
        N, D, M, Ks = 100, 12, 4, 10
        X = np.random.random((N, D)).astype(np.float32)
        pq = nanopq.PQ(M=M, Ks=Ks)
        pq.fit(X)
        X_ = pq.encode(X)  # encoded
        self.assertEqual(X_.shape, (N, M))
        self.assertEqual(X_.dtype, np.uint8)
        X__ = pq.decode(X_)  # reconstructed
        self.assertEqual(X.shape, X__.shape)
        # The original X and the reconstructed X__ should be similar
        self.assertTrue(np.linalg.norm(X - X__) ** 2 / np.linalg.norm(X) ** 2 < 0.1)

    def test_search(self):
        N, D, M, Ks = 100, 12, 4, 10
        X = np.random.random((N, D)).astype(np.float32)
        pq = nanopq.PQ(M=M, Ks=Ks)
        pq.fit(X)
        X_ = pq.encode(X)
        q = X[13]
        dtbl = pq.dtable(q)
        self.assertEqual(dtbl.dtable.shape, (M, Ks))
        dists = dtbl.adist(X_)
        self.assertEqual(len(dists), N)
        self.assertEqual(np.argmin(dists), 13)
        dists2 = pq.dtable(q).adist(X_)  # can be chained
        self.assertAlmostEqual(dists.tolist(), dists2.tolist())

    def test_pickle(self):
        import pickle

        N, D, M, Ks = 100, 12, 4, 10
        X = np.random.random((N, D)).astype(np.float32)
        pq = nanopq.PQ(M=M, Ks=Ks)
        pq.fit(X)
        dumped = pickle.dumps(pq)
        pq2 = pickle.loads(dumped)
        self.assertEqual(
            (pq.M, pq.Ks, pq.verbose, pq.code_dtype, pq.Ds),
            (pq2.M, pq2.Ks, pq2.verbose, pq2.code_dtype, pq2.Ds),
        )
        self.assertTrue(np.allclose(pq.codewords, pq2.codewords))
        self.assertTrue(pq == pq2)

    def test_ip(self):
        N, D, M, Ks = 100, 12, 4, 10
        X = np.random.random((N, D)).astype(np.float32)
        pq = nanopq.PQ(M=M, Ks=Ks, metric="dot")
        pq.fit(X)
        X_ = pq.encode(X)
        q = X[13]
        dist1 = pq.dtable(q).adist(X_)
        dtable = np.empty((pq.M, pq.Ks), dtype=np.float32)
        for m in range(pq.M):
            query_sub = q[m * pq.Ds : (m + 1) * pq.Ds]
            dtable[m, :] = np.matmul(pq.codewords[m], query_sub[None, :].T).sum(axis=-1)
        dist2 = np.sum(dtable[range(M), X_], axis=1)
        self.assertTrue((dist1 == dist2).all())
        self.assertTrue(abs(np.mean(np.matmul(X, q[:, None]).squeeze() - dist1)) < 1e-7)


if __name__ == "__main__":
    unittest.main()
