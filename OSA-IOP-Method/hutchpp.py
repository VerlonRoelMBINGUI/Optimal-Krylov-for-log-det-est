import numpy as np
import scipy.linalg as la


def hutchpp_trace(Fop, n, m, rng=42, dist="rademacher"):
    """
    Hutch++ trace estimator.
    """

    m_s = m // 3
    m_g = m // 3

    rng = np.random.default_rng(rng)

    S = rng.integers(0, 2, (n, m_s)) * 2 - 1
    S = S / np.sqrt(m_s)

    Y = np.column_stack([Fop(S[:, j]) for j in range(m_s)])

    A, _ = la.qr(Y, mode="economic")

    FA = np.column_stack([Fop(A[:, j]) for j in range(A.shape[1])])

    small_trace = np.trace(A.T @ FA)

    corr = 0.0

    for _ in range(m_g):
        z = rng.integers(0, 2, n) * 2 - 1
        z_perp = z - A @ (A.T @ z)

        if la.norm(z_perp) < 1e-12:
            continue

        corr += np.dot(z_perp, Fop(z_perp))

    return float(small_trace + corr / m_g), {
        "m_S": m_s,
        "m_G": m_g,
        "rank": A.shape[1],
    }