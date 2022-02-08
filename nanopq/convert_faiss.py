# Try to import faiss
import importlib.util

spec = importlib.util.find_spec("faiss")
if spec is None:
    pass  # If faiss hasn't been installed. Just skip
else:
    import faiss

import numpy as np

from .pq import PQ
from .opq import OPQ


def nanopq_to_faiss(pq_nanopq):
    """Convert a :class:`nanopq.PQ` instance to `faiss.IndexPQ <https://github.com/facebookresearch/faiss/blob/master/IndexPQ.h>`_.
    To use this function, `faiss module needs to be installed <https://github.com/facebookresearch/faiss/blob/master/INSTALL.md>`_.

    Args:
        pq_nanopq (nanopq.PQ): An input PQ instance.

    Returns:
        faiss.IndexPQ: A converted PQ instance, with the same codewords to the input.

    """
    assert isinstance(pq_nanopq, PQ), "Error. pq_nanopq must be nanopq.pq"
    assert (
        pq_nanopq.codewords is not None
    ), "Error. pq_nanopq.codewords must have been set beforehand"
    D = pq_nanopq.Ds * pq_nanopq.M
    nbits = {np.uint8: 8, np.uint16: 16, np.uint32: 32}[pq_nanopq.code_dtype]

    pq_faiss = faiss.IndexPQ(D, pq_nanopq.M, nbits)

    for m in range(pq_nanopq.M):
        # Prepare std::vector<float>
        codewords_cpp_m = faiss.Float32Vector()

        # Flatten m-th codewords from (Ks, Ds) to (Ks * Ds, ), then copy them to cpp
        faiss.copy_array_to_vector(pq_nanopq.codewords[m].reshape(-1), codewords_cpp_m)

        # Set the codeword to ProductQuantizer in IndexPQ
        pq_faiss.pq.set_params(centroids=codewords_cpp_m.data(), m=m)

    pq_faiss.is_trained = True

    return pq_faiss


def faiss_to_nanopq(pq_faiss):
    """Convert a `faiss.IndexPQ <https://github.com/facebookresearch/faiss/blob/master/IndexPQ.h>`_ instance to :class:`nanopq.PQ`.
    To use this function, `faiss module needs to be installed <https://github.com/facebookresearch/faiss/blob/master/INSTALL.md>`_.

    Args:
        pq_faiss (faiss.IndexPQ): An input PQ instance.

    Returns:
        tuple:
            * nanopq.PQ: A converted PQ instance, with the same codewords to the input.
            * np.ndarray: Stored PQ codes in the input IndexPQ, with the shape=(N, M). This will be empty if codes are not stored

    """
    assert isinstance(pq_faiss, faiss.IndexPQ), "Error. pq_faiss must be IndexPQ"
    assert pq_faiss.is_trained, "Error. pq_faiss must have been trained"

    pq_nanopq = PQ(M=pq_faiss.pq.M, Ks=int(2 ** pq_faiss.pq.nbits))
    pq_nanopq.Ds = int(pq_faiss.pq.d / pq_faiss.pq.M)

    # Extract codewords from pq_IndexPQ.ProductQuantizer, reshape them to M*Ks*Ds
    codewords = faiss.vector_to_array(pq_faiss.pq.centroids).reshape(
        pq_nanopq.M, pq_nanopq.Ks, pq_nanopq.Ds
    )

    pq_nanopq.codewords = codewords

    return pq_nanopq, faiss.vector_to_array(pq_faiss.codes).reshape(-1, pq_faiss.pq.M)


def faiss_to_nanopq_opq(opq_faiss):
    """Convert a `faiss.IndexPreTransform <https://github.com/facebookresearch/faiss/blob/master/IndexPreTransform.h>`_ instance to :class:`nanopq.OPQ`.
    To use this function, `faiss module needs to be installed <https://github.com/facebookresearch/faiss/blob/master/INSTALL.md>`_.

    Args:
        opq_faiss (faiss.IndexPreTransform): An input OPQ instance. It only supports `OPQMatrix + IndexPQ`, not `+ IndexIVFPQ`.

    Returns:
        tuple:
            * nanopq.OPQ: A converted OPQ instance, with the same codewords to the input.
            * np.ndarray: Stored PQ codes in the input IndexPQ, with the shape=(N, M). This will be empty if codes are not stored

    """
    assert isinstance(opq_faiss, faiss.IndexPreTransform), "Error. opq_faiss must be IndexPreTransform"
    assert opq_faiss.is_trained, "Error. opq_faiss must have been trained"

    opq_matrix: faiss.LinearTransform = faiss.downcast_VectorTransform(opq_faiss.chain.at(0))
    pq_index: faiss.IndexPQ = faiss.downcast_index(opq_faiss.index)

    opq_nanopq = OPQ(M=pq_index.pq.M, Ks=int(2 ** pq_index.pq.nbits))
    opq_nanopq.pq.Ds = int(pq_index.pq.d / pq_index.pq.M)

    # Extract codewords from pq_IndexPQ.ProductQuantizer, reshape them to M*Ks*Ds
    codewords = faiss.vector_to_array(pq_index.pq.centroids).reshape(
        opq_nanopq.M, opq_nanopq.Ks, opq_nanopq.Ds
    )

    opq_nanopq.pq.codewords = codewords

    opq_nanopq.R = (
        faiss.vector_to_array(opq_matrix.A)
        .reshape(opq_matrix.d_in, opq_matrix.d_out)
        .transpose(1, 0)
    )

    return opq_nanopq, faiss.vector_to_array(pq_index.codes).reshape(-1, pq_index.pq.M)
