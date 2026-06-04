from typing import Tuple

import numpy as np
import scipy.sparse as sp


def get_focal_interval(Q: sp.spmatrix) -> Tuple[float, float]:
    """
    Estimate the spectral interval of Q using Gershgorin circles.

    Parameters
    ----------
    Q : scipy.sparse.spmatrix
        Sparse square matrix.

    Returns
    -------
    alpha : float
        Lower spectral bound.
    beta : float
        Upper spectral bound.
    """
    D = Q.diagonal()
    R = np.array(Q.sum(axis=1)).ravel() - np.abs(D)

    alpha = np.min(D - R)
    beta = np.max(D + R)

    alpha = max(alpha, 1e-12)

    return float(alpha), float(beta)