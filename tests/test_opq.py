import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
import unittest

import nanopq
import numpy as np


class TestSuite(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)

    def test_property(self):
        opq = nanopq.OPQ(M=4, Ks=256)
        self.assertEqual(
            (opq.M, opq.Ks, opq.verbose, opq.code_dtype),
            (opq.pq.M, opq.pq.Ks, opq.pq.verbose, opq.pq.code_dtype),
        )

    def test_fit(self):
        N, D, M, Ks = 100, 12, 4, 10
        X = np.random.random((N, D)).astype(np.float32)
        opq = nanopq.OPQ(M=M, Ks=Ks)
        opq.fit(X)
        self.assertEqual(opq.Ds, D / M)
        self.assertEqual(opq.codewords.shape, (M, Ks, D / M))
        self.assertEqual(opq.R.shape, (D, D))

        opq2 = nanopq.OPQ(M=M, Ks=Ks).fit(X)  # Can be called as a chain
        self.assertTrue(np.allclose(opq.codewords, opq2.codewords))

    def test_eq(self):
        import copy

        N, D, M, Ks = 100, 12, 4, 10
        X = np.random.random((N, D)).astype(np.float32)
        opq1 = nanopq.OPQ(M=M, Ks=Ks)
        opq2 = nanopq.OPQ(M=M, Ks=Ks)
        opq3 = copy.deepcopy(opq1)
        opq4 = nanopq.OPQ(M=M, Ks=2 * Ks)
        self.assertTrue(opq1 == opq1)
        self.assertTrue(opq1 == opq2)
        self.assertTrue(opq1 == opq3)
        self.assertTrue(opq1 != opq4)

        opq1.fit(X)
        opq2.fit(X)
        opq3 = copy.deepcopy(opq1)
        opq4.fit(X)
        self.assertTrue(opq1 == opq1)
        self.assertTrue(opq1 == opq2)
        self.assertTrue(opq1 == opq3)
        self.assertTrue(opq1 != opq4)

    def test_rotate(self):
        N, D, M, Ks = 100, 12, 4, 10
        X = np.random.random((N, D)).astype(np.float32)
        opq = nanopq.OPQ(M=M, Ks=Ks)
        opq.fit(X)
        rotated_vec = opq.rotate(X[0])
        rotated_vecs = opq.rotate(X[:3])
        self.assertEqual(rotated_vec.shape, (D,))
        self.assertEqual(rotated_vecs.shape, (3, D))

        # Because R is a rotation matrix (R^t * R = I), R^t should be R^(-1)
        self.assertAlmostEqual(
            np.linalg.norm(opq.R.T - np.linalg.inv(opq.R)), 0.0, places=3
        )

    def test_parametric_init(self):
        N, D, M, Ks = 100, 12, 2, 20
        X = np.random.random((N, D)).astype(np.float32)
        opq = nanopq.OPQ(M=M, Ks=Ks)
        opq.fit(X, parametric_init=False, rotation_iter=1)
        err_init = np.linalg.norm(X - opq.decode(opq.encode(X)))

        opq = nanopq.OPQ(M=M, Ks=Ks)
        opq.fit(X, parametric_init=True, rotation_iter=1)
        err = np.linalg.norm(X - opq.decode(opq.encode(X)))

        self.assertLess(err, err_init)


if __name__ == "__main__":
    unittest.main()
