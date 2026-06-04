import numpy as np
from numpy.linalg import eigh


def lanczos_tridiag(Aop, v, m, tol=1e-10):
    v = np.asarray(v, dtype=float).copy()

    beta_prev = 0.0
    v0 = np.zeros_like(v)

    v = v / (np.linalg.norm(v) + 1e-32)

    Vcols = []
    alphas, betas = [], []

    for _ in range(m):
        w = np.asarray(Aop.matvec(v), dtype=float)

        alpha = float(np.dot(v, w))
        w = w - alpha * v - beta_prev * v0
        beta = np.linalg.norm(w)

        Vcols.append(v.copy())
        alphas.append(alpha)

        if beta < tol:
            break

        if len(Vcols) > 1:
            betas.append(beta_prev)

        v0, v = v, w / (beta + 1e-32)
        beta_prev = beta

    V = np.stack(Vcols, axis=1)
    T = np.diag(alphas)

    if len(Vcols) > 1:
        off = np.array(betas[:len(Vcols) - 1])
        T += np.diag(off, 1) + np.diag(off, -1)

    return V, T


def quadratic_form_log_psd(Aop, v, m, alpha):
    V, T = lanczos_tridiag(Aop, v, m)

    evals, U = eigh(T)
    evals = np.maximum(evals, 1e-300)

    scaled = evals / alpha
    log_vals = -np.log(scaled)

    weights = U[0, :] ** 2

    return float(np.dot(weights, log_vals) * (np.linalg.norm(v) ** 2))