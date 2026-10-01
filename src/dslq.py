import time
import numpy as np
import scipy.io
from numpy.linalg import qr
from scipy.sparse import issparse
from scipy.sparse.linalg import LinearOperator


# ============================================================
# Utilities
# ============================================================

def as_linear_operator(Q):
    """Return Q as a scipy LinearOperator."""
    if isinstance(Q, LinearOperator):
        return Q

    if issparse(Q):
        Q = Q.tocsr()
        n = Q.shape[0]
        return LinearOperator(
            (n, n),
            matvec=lambda x: Q @ x,
            rmatvec=lambda x: Q.T @ x,
            matmat=lambda X: Q @ X,
            rmatmat=lambda X: Q.T @ X,
            dtype=float,
        )

    Q = np.asarray(Q, dtype=float)
    if Q.ndim != 2:
        raise ValueError(f"Q must be a 2D matrix, got shape {Q.shape}")

    n = Q.shape[0]
    return LinearOperator(
        (n, n),
        matvec=lambda x: Q @ x,
        rmatvec=lambda x: Q.T @ x,
        matmat=lambda X: Q @ X,
        rmatmat=lambda X: Q.T @ X,
        dtype=float,
    )


def gaussian_matrix(n, s, rng):
    return rng.standard_normal((n, s))


def rademacher_matrix(n, s, rng):
    return (2 * rng.integers(0, 2, size=(n, s)) - 1).astype(float)


def get_gershgorin_interval(Q):
    """
    Gershgorin interval for an explicitly stored sparse/dense matrix.

    This spectral setup time is included in the reported total runtime,
    but it is not charged to the 1800 logdet matvec budget.
    """
    if isinstance(Q, LinearOperator):
        raise ValueError(
            "Gershgorin interval requires an explicitly stored matrix."
        )

    D = np.asarray(Q.diagonal(), dtype=float).ravel()

    if issparse(Q):
        R = np.asarray(np.abs(Q).sum(axis=1)).ravel() - np.abs(D)
    else:
        R = np.sum(np.abs(Q), axis=1) - np.abs(D)

    lambda_min = float(np.min(D - R))
    lambda_max = float(np.max(D + R))

    if lambda_min <= 0.0:
        lambda_min = max(lambda_min, 1e-12)

    return lambda_min, lambda_max


def load_suite_sparse_matrix(file_path, matrix_name="A"):
    """Load a SuiteSparse-style .mat matrix."""
    data = scipy.io.loadmat(file_path)

    if "Problem" in data:
        problem = data["Problem"]
        if problem.dtype.names is None or matrix_name not in problem.dtype.names:
            raise KeyError(
                f"Field '{matrix_name}' not found in Problem. "
                f"Available fields: {problem.dtype.names}"
            )
        Q = problem[matrix_name][0, 0]
    elif matrix_name in data:
        Q = data[matrix_name]
    else:
        raise KeyError(
            f"Could not find Problem['{matrix_name}'] or '{matrix_name}' "
            f"in {file_path}."
        )

    if issparse(Q):
        return Q.tocsr()

    from scipy.sparse import csr_matrix
    return csr_matrix(Q)


# ============================================================
# Scaling J = Q / alpha
# ============================================================

def normalized_operator(Q, lambda_min):
    """
    Same scaling used in the previous implementation:
        alpha = lambda_min if lambda_min < 1,
                1 otherwise,
        J = Q / alpha.
    """
    if lambda_min <= 0:
        raise ValueError("lambda_min must be positive.")

    Qop = as_linear_operator(Q)
    n = Qop.shape[0]

    alpha = lambda_min if lambda_min < 1.0 else 1.0
    scaled = alpha != 1.0

    if scaled:
        Jop = LinearOperator(
            (n, n),
            matvec=lambda x: Qop.matvec(x) / alpha,
            rmatvec=lambda x: Qop.rmatvec(x) / alpha,
            matmat=lambda X: Qop.matmat(X) / alpha,
            rmatmat=lambda X: Qop.rmatmat(X) / alpha,
            dtype=float,
        )
    else:
        Jop = Qop

    return Jop, alpha, scaled


def shifted_operator(Jop, gamma):
    """
    K = J - gamma I.
    The algorithm in the manuscript uses gamma in {0,1}.
    """
    if gamma not in (0.0, 1.0):
        raise ValueError("To follow the manuscript exactly, gamma must be 0 or 1.")

    n = Jop.shape[0]

    return LinearOperator(
        (n, n),
        matvec=lambda x: Jop.matvec(x) - gamma * x,
        rmatvec=lambda x: Jop.rmatvec(x) - gamma * x,
        matmat=lambda X: Jop.matmat(X) - gamma * X,
        rmatmat=lambda X: Jop.rmatmat(X) - gamma * X,
        dtype=float,
    )


# ============================================================
# Batched independent Lanczos recurrences
# ============================================================

def block_lanczos_tridiag(Aop, V, m, tol=1e-12):
    """
    Run independent m-step Lanczos recurrences for all columns of V,
    batched through Aop.matmat.

    This is mathematically equivalent to the FOR loops in the algorithm.
    """
    if m < 1:
        raise ValueError("Lanczos degree m must be >= 1.")

    n, N = V.shape

    norm_v2 = np.einsum("ij,ij->j", V, V)
    norms = np.sqrt(np.maximum(norm_v2, 1e-300))

    Vcur = V / norms[None, :]
    Vprev = np.zeros_like(Vcur)
    beta_prev = np.zeros(N)

    alphas = np.empty((m, N), dtype=float)
    betas = np.empty((max(m - 1, 0), N), dtype=float)

    for j in range(m):
        W = Aop.matmat(Vcur)

        alpha_j = np.einsum("ij,ij->j", Vcur, W)

        W = W - Vcur * alpha_j[None, :]
        W = W - Vprev * beta_prev[None, :]

        beta_j = np.sqrt(
            np.maximum(np.einsum("ij,ij->j", W, W), 0.0)
        )

        alphas[j, :] = alpha_j

        if j < m - 1:
            betas[j, :] = beta_j

        beta_safe = np.where(beta_j > tol, beta_j, 1.0)

        Vprev, Vcur = Vcur, W / beta_safe[None, :]
        beta_prev = beta_j

    return alphas, betas, norm_v2


def batch_slq_log_quadratic_forms(Aop, V, m):
    """
    Approximate v_j^T log(A) v_j for each column v_j of V.
    """
    alphas, betas, norm_v2 = block_lanczos_tridiag(Aop, V, m)

    m_eff, N = alphas.shape

    T = np.zeros((N, m_eff, m_eff), dtype=float)

    idx = np.arange(m_eff)
    T[:, idx, idx] = alphas.T

    if m_eff > 1:
        off = np.arange(m_eff - 1)
        T[:, off, off + 1] = betas.T
        T[:, off + 1, off] = betas.T

    theta, X = np.linalg.eigh(T)

    # J is SPD; clipping only protects np.log against tiny negative
    # Ritz values caused by roundoff.
    theta = np.maximum(theta, 1e-300)

    # tau_k = e_1^T x_k
    tau2 = X[:, 0, :] ** 2

    quadrature = np.sum(tau2 * np.log(theta), axis=1)

    return norm_v2 * quadrature


# ============================================================
#  DSLQ structure
# ============================================================

def dslq_logdet(
    Q,
    mvec=30,
    m=89,
    m_prime=90,
    gamma=1,
    sketch="gaussian",
    lambda_min=None,
    seed=42,
    return_breakdown=False,
):
    """
    DSLQ estimator 

    Manuscript structure:
      1) s = mvec/3 sketch vectors
      2) Y = (J - gamma I) S
      3) QR -> A
      4) s projected Lanczos runs of degree m
      5) r = mvec/3 residual Rademacher probes
      6) residual Lanczos runs of degree m_prime
      7) residual average applied AFTER accumulation
      8) n log(alpha) scaling correction
    """
    if mvec % 3 != 0:
        raise ValueError("mvec must be divisible by 3.")

    if gamma not in (0, 1):
        raise ValueError("gamma must be 0 or 1 to follow the algorithm.")

    rng = np.random.default_rng(seed)

    Qop = as_linear_operator(Q)
    n = Qop.shape[0]

    s = mvec // 3
    r = mvec // 3

    # --------------------------------------------------------
    # Spectral setup
    # --------------------------------------------------------
    spectral_start = time.perf_counter()

    if lambda_min is None:
        lambda_min, lambda_max = get_gershgorin_interval(Q)
    else:
        _, lambda_max = get_gershgorin_interval(Q)

    Jop, alpha, scaled = normalized_operator(Q, lambda_min)

    spectral_time = time.perf_counter() - spectral_start

    estimator_start = time.perf_counter()

    # --------------------------------------------------------
    # Step 1: sketch S
    # --------------------------------------------------------
    if sketch.lower() == "gaussian":
        S = gaussian_matrix(n, s, rng)
    elif sketch.lower() == "rademacher":
        S = rademacher_matrix(n, s, rng)
    else:
        raise ValueError("sketch must be 'gaussian' or 'rademacher'.")

    # --------------------------------------------------------
    # Step 2: Y = (J - gamma I) S
    # --------------------------------------------------------
    Kop = shifted_operator(Jop, gamma=float(gamma))
    Y = Kop.matmat(S)

    # --------------------------------------------------------
    # Step 3: QR basis A
    # --------------------------------------------------------
    A, _ = qr(Y, mode="reduced")
    A = A[:, :s]

    # --------------------------------------------------------
    # Step 4: projected term
    #
    # sum_i q_i^T log(J) q_i
    # --------------------------------------------------------
    projected_qforms = batch_slq_log_quadratic_forms(
        Jop, A, m
    )
    trace_projected = float(np.sum(projected_qforms))

    # --------------------------------------------------------
    # Step 5: residual probes
    # --------------------------------------------------------
    V = rademacher_matrix(n, r, rng)

    # W = (I - A A^T)V
    W = V - A @ (A.T @ V)

    norms2 = np.einsum("ij,ij->j", W, W)
    keep = norms2 > 1e-20

    successful = int(np.sum(keep))

    # --------------------------------------------------------
    # Step 6: residual Lanczos + accumulation
    #
    # IMPORTANT:
    # We first accumulate all residual quadratic forms, then
    # divide by the number of successful probes.
    # --------------------------------------------------------
    if successful > 0:
        residual_qforms = batch_slq_log_quadratic_forms(
            Jop, W[:, keep], m_prime
        )

        trace_residual_sum = float(np.sum(residual_qforms))

        # equivalent to 3/mvec when all r=mvec/3 probes survive
        trace_residual = trace_residual_sum / successful
    else:
        trace_residual_sum = 0.0
        trace_residual = 0.0

    # --------------------------------------------------------
    # Step 7: trace(log J)
    # --------------------------------------------------------
    trace_log_J = trace_projected + trace_residual

    # --------------------------------------------------------
    # Step 8: logdet(Q)
    # --------------------------------------------------------
    scaling_correction = n * np.log(alpha) if scaled else 0.0

    estimate = scaling_correction + trace_log_J

    estimator_time = time.perf_counter() - estimator_start

    # Spectral setup is INCLUDED in total computation time.
    total_computation_time = spectral_time + estimator_time

    # --------------------------------------------------------
    # Matvec accounting
    # --------------------------------------------------------
    #
    # Y=(J-gamma I)S        -> s applications of J
    # projected SLQ         -> s*m
    # residual SLQ          -> r*m_prime
    #
    range_matvecs = s
    projected_matvecs = s * m
    residual_matvecs = r * m_prime

    total_logdet_matvecs = (
        range_matvecs
        + projected_matvecs
        + residual_matvecs
    )

    breakdown = {
        "n": n,
        "mvec": mvec,
        "s": s,
        "r": r,
        "m": m,
        "m_prime": m_prime,
        "gamma": gamma,
        "sketch": sketch,
        "lambda_min": lambda_min,
        "lambda_max": lambda_max,
        "alpha": alpha,
        "scaled": scaled,
        "trace_projected": trace_projected,
        "trace_residual_sum": trace_residual_sum,
        "trace_residual": trace_residual,
        "trace_log_J": trace_log_J,
        "scaling_correction": scaling_correction,
        "successful_residual_probes": successful,
        "range_matvecs": range_matvecs,
        "projected_matvecs": projected_matvecs,
        "residual_matvecs": residual_matvecs,
        "total_logdet_matvecs": total_logdet_matvecs,
        "spectral_time": spectral_time,
        "estimator_time": estimator_time,
        "total_computation_time": total_computation_time,
    }

    if return_breakdown:
        return estimate, breakdown

    return estimate


# ============================================================
# ecology2 experiment with EXACT 1800 logdet matvecs
# ============================================================

def run_ecology2(
    file_path="ecology2.mat",
    target_budget=1800,
    mvec=30,
    gamma=1,
    sketch="gaussian",
    seed=42,
    reference_logdet=None,
):
    """
    With mvec=30:
        s = r = 10.

    Choose:
        m       = 89
        m_prime = 90

    Therefore:
        range      = 10
        projected  = 10*89 = 890
        residual   = 10*90 = 900
        total      = 1800 exactly.
    """
    full_start = time.perf_counter()

    Q = load_suite_sparse_matrix(file_path)

    print("=" * 78)
    print("ecology2 -- DSLQ")
    print("=" * 78)
    print(f"Matrix shape                  : {Q.shape}")
    print(f"nnz                           : {Q.nnz}")
    print(f"mvec                          : {mvec}")
    print(f"gamma                         : {gamma}")
    print(f"range sketch                  : {sketch}")
    print(f"target logdet matvec budget   : {target_budget}")

    if mvec != 30 or target_budget != 1800:
        raise ValueError(
            "This experiment is configured for mvec=30 and target_budget=1800. "
            "For other values, choose m and m_prime consistently."
        )

    # Exact budget:
    # 10 + 10*89 + 10*90 = 1800
    m = 89
    m_prime = 90

    estimate, info = dslq_logdet(
        Q,
        mvec=mvec,
        m=m,
        m_prime=m_prime,
        gamma=gamma,
        sketch=sketch,
        lambda_min=None,
        seed=seed,
        return_breakdown=True,
    )

    wall_time = time.perf_counter() - full_start

    print()
    print("-" * 78)
    print("Spectral setup")
    print("-" * 78)
    print(f"lambda_min                    : {info['lambda_min']:}")
    print(f"lambda_max                    : {info['lambda_max']:}")
    print(f"alpha                         : {info['alpha']:}")
    print(f"spectral setup time           : {info['spectral_time']:} s")

    print()
    print("-" * 78)
    print("DSLQ parameters")
    print("-" * 78)
    print(f"s = mvec/3                    : {info['s']}")
    print(f"r = mvec/3                    : {info['r']}")
    print(f"projected Lanczos degree m    : {info['m']}")
    print(f"residual Lanczos degree m'    : {info['m_prime']}")
    print(f"successful residual probes    : {info['successful_residual_probes']}")

    print()
    print("-" * 78)
    print("Matvec budget")
    print("-" * 78)
    print(f"range finder                  : {info['range_matvecs']}")
    print(f"projected SLQ                 : {info['projected_matvecs']}")
    print(f"residual SLQ                  : {info['residual_matvecs']}")
    print(f"TOTAL logdet matvecs          : {info['total_logdet_matvecs']}")
    print(f"target budget                 : {target_budget}")

    if info["total_logdet_matvecs"] != target_budget:
        raise RuntimeError(
            f"Budget mismatch: used {info['total_logdet_matvecs']} "
            f"instead of {target_budget}."
        )

    print()
    print("-" * 78)
    print("Trace decomposition")
    print("-" * 78)
    print(f"projected contribution        : {info['trace_projected']:}")
    print(f"residual contribution         : {info['trace_residual']:}")
    print(f"trace(log(J))                 : {info['trace_log_J']:}")
    print(f"n*log(alpha)                  : {info['scaling_correction']:}")

    print()
    print("=" * 78)
    print("FINAL RESULT")
    print("=" * 78)
    print(f"Estimated logdet(Q)           : {estimate:}")
    print(f"Estimator time                : {info['estimator_time']:} s")
    print(f"Spectral setup time           : {info['spectral_time']:} s")
    print(f"TOTAL computation time        : {info['total_computation_time']:} s")
    print(f"TOTAL wall-clock time         : {wall_time:} s")

    # if reference_logdet is not None:
    #     abs_error = abs(estimate - reference_logdet)
    #     rel_error = abs_error / max(abs(reference_logdet), 1e-300)

    #     print(f"Reference logdet              : {reference_logdet:.12e}")
    #     print(f"Absolute error                : {abs_error:.12e}")
    #     print(f"Relative error                : {rel_error:.12e}")

    print("=" * 78)

    return estimate, info
