import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
import importlib.util
import unittest

import nanopq
import numpy as np

spec = importlib.util.find_spec("faiss")
if spec is None:
    raise unittest.SkipTest(
        "Cannot find the faiss module. Skipt the test for convert_faiss"
    )
else:
    import faiss

    print("faiss version:", faiss.__version__)


class TestSuite(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)

    def test_nanopq_to_faiss(self):
        D, M, Ks = 32, 4, 256
        Nt, Nb, Nq = 2000, 10000, 100
        Xt = np.random.rand(Nt, D).astype(np.float32)
        Xb = np.random.rand(Nb, D).astype(np.float32)
        Xq = np.random.rand(Nq, D).astype(np.float32)
        pq_nanopq = nanopq.PQ(M=M, Ks=Ks)
        pq_nanopq.fit(vecs=Xt)

        with self.assertRaises(AssertionError):  # opq is not supported
            opq = nanopq.OPQ(M=M, Ks=Ks)
            nanopq.nanopq_to_faiss(opq)

        pq_faiss = nanopq.nanopq_to_faiss(pq_nanopq)  # IndexPQ

        # Encoded results should be same
        Cb_nanopq = pq_nanopq.encode(vecs=Xb)
        Cb_faiss = pq_faiss.pq.compute_codes(x=Xb)  # ProductQuantizer in IndexPQ
        self.assertTrue(np.array_equal(Cb_nanopq, Cb_faiss))

        # Search result should be same
        topk = 10
        pq_faiss.add(Xb)
        _, ids1 = pq_faiss.search(x=Xq, k=topk)
        ids2 = np.array(
            [
                np.argsort(pq_nanopq.dtable(query=xq).adist(codes=Cb_nanopq))[:topk]
                for xq in Xq
            ]
        )

        self.assertTrue(np.array_equal(ids1, ids2))

    def test_faiss_to_nanopq_pq(self):
        D, M, Ks = 32, 4, 256
        Nt, Nb, Nq = 2000, 10000, 100
        nbits = int(np.log2(Ks))
        assert nbits == 8
        Xt = np.random.rand(Nt, D).astype(np.float32)
        Xb = np.random.rand(Nb, D).astype(np.float32)
        Xq = np.random.rand(Nq, D).astype(np.float32)

        pq_faiss = faiss.IndexPQ(D, M, nbits)
        pq_faiss.train(x=Xt)
        pq_faiss.add(x=Xb)

        pq_nanopq, Cb_faiss = nanopq.faiss_to_nanopq(pq_faiss=pq_faiss)
        self.assertIsInstance(pq_nanopq, nanopq.PQ)
        self.assertEqual(pq_nanopq.codewords.shape, (M, Ks, int(D / M)))

        # Encoded results should be same
        Cb_nanopq = pq_nanopq.encode(vecs=Xb)
        self.assertTrue(np.array_equal(Cb_nanopq, Cb_faiss))

        # Search result should be same
        topk = 100
        _, ids1 = pq_faiss.search(x=Xq, k=topk)
        ids2 = np.array(
            [
                np.argsort(pq_nanopq.dtable(query=xq).adist(codes=Cb_nanopq))[:topk]
                for xq in Xq
            ]
        )
        self.assertTrue(np.array_equal(ids1, ids2))

    def test_faiss_to_nanopq_opq(self):
        D, M, Ks = 32, 4, 256
        Nt, Nb, Nq = 2000, 10000, 100
        nbits = int(np.log2(Ks))
        assert nbits == 8
        Xt = np.random.rand(Nt, D).astype(np.float32)
        Xb = np.random.rand(Nb, D).astype(np.float32)
        Xq = np.random.rand(Nq, D).astype(np.float32)

        pq_faiss = faiss.IndexPQ(D, M, nbits)
        opq_matrix = faiss.OPQMatrix(D, M=M)
        pq_faiss = faiss.IndexPreTransform(opq_matrix, pq_faiss)
        pq_faiss.train(x=Xt)
        pq_faiss.add(x=Xb)

        pq_nanopq, Cb_faiss = nanopq.faiss_to_nanopq(pq_faiss=pq_faiss)
        self.assertIsInstance(pq_nanopq, nanopq.OPQ)
        self.assertEqual(pq_nanopq.codewords.shape, (M, Ks, int(D / M)))

        # Encoded results should be same
        Cb_nanopq = pq_nanopq.encode(vecs=Xb)
        self.assertTrue(np.array_equal(Cb_nanopq, Cb_faiss))

        # Search result should be same
        topk = 100
        _, ids1 = pq_faiss.search(x=Xq, k=topk)
        ids2 = np.array(
            [
                np.argsort(pq_nanopq.dtable(query=xq).adist(codes=Cb_nanopq))[:topk]
                for xq in Xq
            ]
        )
        self.assertTrue(np.array_equal(ids1, ids2))

    def test_faiss_nanopq_compare_accuracy(self):
        D, M, Ks = 32, 4, 256
        Nt, Nb, Nq = 20000, 10000, 100
        nbits = int(np.log2(Ks))
        assert nbits == 8
        Xt = np.random.rand(Nt, D).astype(np.float32)
        Xb = np.random.rand(Nb, D).astype(np.float32)
        Xq = np.random.rand(Nq, D).astype(np.float32)

        pq_faiss = faiss.IndexPQ(D, M, nbits)
        pq_faiss.train(x=Xt)
        Cb_faiss = pq_faiss.pq.compute_codes(Xb)
        Xb_faiss_ = pq_faiss.pq.decode(Cb_faiss)

        pq_nanopq = nanopq.PQ(M=M, Ks=Ks)
        pq_nanopq.fit(vecs=Xt)
        Cb_nanopq = pq_nanopq.encode(vecs=Xb)
        Xb_nanopq_ = pq_nanopq.decode(codes=Cb_nanopq)

        # Reconstruction error should be almost identical
        avg_relative_error_faiss = ((Xb - Xb_faiss_) ** 2).sum() / (Xb**2).sum()
        avg_relative_error_nanopq = ((Xb - Xb_nanopq_) ** 2).sum() / (Xb**2).sum()
        diff_rel = (
            avg_relative_error_faiss - avg_relative_error_nanopq
        ) / avg_relative_error_faiss
        diff_rel = np.sqrt(diff_rel**2)
        print("avg_rel_error_faiss:", avg_relative_error_faiss)
        print("avg_rel_error_nanopq:", avg_relative_error_nanopq)
        print("diff rel:", diff_rel)

        self.assertLess(diff_rel, 0.01)


if __name__ == "__main__":
    unittest.main()
