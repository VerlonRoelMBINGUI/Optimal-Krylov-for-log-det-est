import numpy as np


def gaussian_matrix(n, s, rng):
    return rng.standard_normal((n, s))


def rademacher(n, rng):
    return (rng.integers(0, 2, size=n) * 2 - 1).astype(float)


def gershgorin_interval(Q):
    """
    Compute spectral bounds [lambda_min, lambda_max].
    """
    D = Q.diagonal()
    R = np.array(Q.sum(axis=1)).flatten() - np.abs(D)

    lam_min = np.min(D - R)
    lam_max = np.max(D + R)

    if lam_min < 0:
        lam_min = max(lam_min, 1e-12)

    return lam_min, lam_max