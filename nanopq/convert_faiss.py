# Try to import faiss
import importlib.util

spec = importlib.util.find_spec("faiss")
if spec is None:
    pass  # If faiss hasn't been installed. Just skip
else:
    import faiss
    faiss_metric_map = {
        "l2": faiss.METRIC_L2,
        "dot": faiss.METRIC_INNER_PRODUCT,
        "angular": faiss.METRIC_INNER_PRODUCT,
    }


import numpy as np

from .opq import OPQ
from .pq import PQ



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

    pq_faiss = faiss.IndexPQ(D, pq_nanopq.M, nbits, faiss_metric_map[pq_nanopq.metric])

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
    """Convert a `faiss.IndexPQ <https://github.com/facebookresearch/faiss/blob/master/IndexPQ.h>`_
    or a `faiss.IndexPreTransform <https://github.com/facebookresearch/faiss/blob/master/IndexPreTransform.h>`_ instance to :class:`nanopq.OPQ`.
    To use this function, `faiss module needs to be installed <https://github.com/facebookresearch/faiss/blob/master/INSTALL.md>`_.

    Args:
        pq_faiss (Union[faiss.IndexPQ, faiss.IndexPreTransform]): An input PQ or OPQ instance.

    Returns:
        tuple:
            * Union[nanopq.PQ, nanopq.OPQ]: A converted PQ or OPQ instance, with the same codewords to the input.
            * np.ndarray: Stored PQ codes in the input IndexPQ, with the shape=(N, M). This will be empty if codes are not stored

    """
    assert isinstance(
        pq_faiss, (faiss.IndexPQ, faiss.IndexPreTransform)
    ), "Error. pq_faiss must be IndexPQ or IndexPreTransform"
    assert pq_faiss.is_trained, "Error. pq_faiss must have been trained"

    if isinstance(pq_faiss, faiss.IndexPreTransform):
        opq_matrix: faiss.LinearTransform = faiss.downcast_VectorTransform(
            pq_faiss.chain.at(0)
        )
        pq_faiss: faiss.IndexPQ = faiss.downcast_index(pq_faiss.index)
        pq_nanopq = OPQ(M=pq_faiss.pq.M, Ks=int(2**pq_faiss.pq.nbits))
        pq_nanopq.pq.Ds = int(pq_faiss.pq.d / pq_faiss.pq.M)

        # Extract codewords from pq_IndexPQ.ProductQuantizer, reshape them to M*Ks*Ds
        codewords = faiss.vector_to_array(pq_faiss.pq.centroids).reshape(
            pq_nanopq.M, pq_nanopq.Ks, pq_nanopq.Ds
        )

        pq_nanopq.pq.codewords = codewords
        pq_nanopq.R = (
            faiss.vector_to_array(opq_matrix.A)
            .reshape(opq_matrix.d_out, opq_matrix.d_in)
            .transpose(1, 0)
        )
    else:
        pq_nanopq = PQ(M=pq_faiss.pq.M, Ks=int(2**pq_faiss.pq.nbits))
        pq_nanopq.Ds = int(pq_faiss.pq.d / pq_faiss.pq.M)

        # Extract codewords from pq_IndexPQ.ProductQuantizer, reshape them to M*Ks*Ds
        codewords = faiss.vector_to_array(pq_faiss.pq.centroids).reshape(
            pq_nanopq.M, pq_nanopq.Ks, pq_nanopq.Ds
        )
        pq_nanopq.codewords = codewords

    return pq_nanopq, faiss.vector_to_array(pq_faiss.codes).reshape(-1, pq_faiss.pq.M)
