import numpy as np
from scipy.sparse.linalg import LinearOperator
from scipy.sparse import issparse


class SymmetricOperator:
    """
    Wraps dense/sparse matrices into a unified LinearOperator interface.
    """

    def __init__(self, Q):
        self.Q_raw = Q
        self.op = self._to_operator(Q)
        self.n = self.op.shape[0]

    def _to_operator(self, Q):
        if isinstance(Q, LinearOperator):
            return Q

        if issparse(Q):
            Q = Q.tocsr()
            n = Q.shape[0]

            return LinearOperator(
                (n, n),
                matvec=lambda x: np.asarray(Q @ x, dtype=float),
                rmatvec=lambda x: np.asarray(Q.T @ x, dtype=float),
                dtype=float,
            )

        Q = np.asarray(Q)
        n = Q.shape[0]

        return LinearOperator(
            (n, n),
            matvec=lambda x: np.asarray(Q @ x, dtype=float),
            rmatvec=lambda x: np.asarray(Q.T @ x, dtype=float),
            dtype=float,
        )