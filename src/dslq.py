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
# 10-TRIAL ecology2 EXPERIMENT
# ============================================================

def run_ecology2_10_trials(
    file_path="ecology2.mat",
    target_budget=1800,
    mvec=30,
    gamma=1,
    sketch="gaussian",
    seeds=range(1, 11),
    reference_logdet=None,
):
    """
    Run DSLQ independently for 10 random seeds and report the
    average estimated log-determinant.

    Each trial uses exactly 1800 logdet matvecs.

    Parameters
    ----------
    file_path : str
        Path to the SuiteSparse matrix.

    target_budget : int
        Logdet matvec budget PER TRIAL.

    mvec : int
        Total probe parameter. For mvec=30:
            s = 10,
            r = 10.

    gamma : {0, 1}
        Shift used in the randomized range construction.

    sketch : {"gaussian", "rademacher"}
        Distribution used for the range sketch.

    seeds : iterable
        Independent random seeds.

    reference_logdet : float or None
        Exact/reference logdet if available.
    """

    # --------------------------------------------------------
    # Load matrix ONCE
    # --------------------------------------------------------
    Q = load_suite_sparse_matrix(file_path)

    print("=" * 78)
    print("ecology2 -- DSLQ -- 10 independent trials")
    print("=" * 78)
    print(f"Matrix shape                  : {Q.shape}")
    print(f"nnz                           : {Q.nnz}")
    print(f"mvec                          : {mvec}")
    print(f"gamma                         : {gamma}")
    print(f"range sketch                  : {sketch}")
    print(f"target matvecs / trial        : {target_budget}")
    print(f"number of trials              : {len(list(seeds))}")

    # --------------------------------------------------------
    # Exact 1800-MV configuration
    # --------------------------------------------------------
    if mvec != 30 or target_budget != 1800:
        raise ValueError(
            "This experiment is configured for "
            "mvec=30 and target_budget=1800."
        )

    # 10 + 10*89 + 10*90 = 1800
    m = 89
    m_prime = 90

    # Convert seeds to a fixed list
    seeds = list(seeds)

    # --------------------------------------------------------
    # IMPORTANT:
    # Compute Gershgorin interval ONCE
    #
    # It is deterministic, so there is no reason to recompute
    # it for every stochastic trial.
    # --------------------------------------------------------
    spectral_start = time.perf_counter()

    lambda_min, lambda_max = get_gershgorin_interval(Q)

    spectral_time_once = time.perf_counter() - spectral_start

    print()
    print("-" * 78)
    print("Spectral setup")
    print("-" * 78)
    print(f"lambda_min                    : {lambda_min:.12e}")
    print(f"lambda_max                    : {lambda_max:.12e}")
    print(f"spectral setup time           : {spectral_time_once:.6f} s")

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------
    estimates = []
    estimator_times = []
    trial_total_times = []
    matvecs = []

    if reference_logdet is not None:
        absolute_errors = []
        relative_errors = []

    # --------------------------------------------------------
    # 10 independent stochastic trials
    # --------------------------------------------------------
    print()
    print("=" * 78)
    print("INDIVIDUAL TRIALS")
    print("=" * 78)

    for trial_id, seed in enumerate(seeds, start=1):

        trial_start = time.perf_counter()

        estimate, info = oslq_logdet(
            Q,
            mvec=mvec,
            m=m,
            m_prime=m_prime,
            gamma=gamma,
            sketch=sketch,

            # Reuse the same deterministic spectral estimate
            lambda_min=lambda_min,

            # Different random realization
            seed=seed,

            return_breakdown=True,
        )

        trial_wall_time = time.perf_counter() - trial_start

        # ----------------------------------------------------
        # Verify budget
        # ----------------------------------------------------
        if info["total_logdet_matvecs"] != target_budget:
            raise RuntimeError(
                f"Trial {trial_id}: budget mismatch. "
                f"Used {info['total_logdet_matvecs']} "
                f"instead of {target_budget}."
            )

        estimates.append(estimate)
        estimator_times.append(info["estimator_time"])
        trial_total_times.append(trial_wall_time)
        matvecs.append(info["total_logdet_matvecs"])

        print(
            f"Trial {trial_id:2d} | "
            f"seed = {seed:3d} | "
            f"logdet = {estimate:.12e} | "
            f"time = {info['estimator_time']:.4f} s"
        )

        if reference_logdet is not None:
            abs_error = abs(estimate - reference_logdet)
            rel_error = abs_error / max(
                abs(reference_logdet), 1e-300
            )

            absolute_errors.append(abs_error)
            relative_errors.append(rel_error)

            print(
                f"         absolute error = {abs_error:.6e} | "
                f"relative error = {rel_error:.6e}"
            )

    # --------------------------------------------------------
    # Convert to NumPy arrays
    # --------------------------------------------------------
    estimates = np.asarray(estimates, dtype=float)
    estimator_times = np.asarray(estimator_times, dtype=float)
    trial_total_times = np.asarray(trial_total_times, dtype=float)
    matvecs = np.asarray(matvecs, dtype=int)

    # --------------------------------------------------------
    # Statistics of the logdet estimates
    # --------------------------------------------------------
    mean_estimate = float(np.mean(estimates))
    std_estimate = float(np.std(estimates, ddof=1))
    median_estimate = float(np.median(estimates))

    q1_estimate = float(np.percentile(estimates, 25))
    q3_estimate = float(np.percentile(estimates, 75))
    iqr_estimate = q3_estimate - q1_estimate

    mean_estimator_time = float(np.mean(estimator_times))
    std_estimator_time = float(np.std(estimator_times, ddof=1))

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------
    print()
    print("=" * 78)
    print("10-TRIAL SUMMARY")
    print("=" * 78)

    print(f"Trials                        : {len(estimates)}")
    print(f"Seeds                         : {seeds}")
    print(f"Matvecs per trial             : {target_budget}")
    print(
        f"Total matvecs over 10 trials : "
        f"{int(np.sum(matvecs))}"
    )

    print()
    print("Log-determinant estimates")
    print("-" * 78)

    for i, (seed, est) in enumerate(zip(seeds, estimates), start=1):
        print(
            f"trial {i:2d}, seed {seed:3d} : "
            f"{est:.12e}"
        )

    print()
    print(f"Mean estimated logdet         : {mean_estimate:}")
    print(f"Std. deviation                : {std_estimate:.12e}")
    print(f"Median estimated logdet       : {median_estimate:.12e}")
    print(f"IQR                            : {iqr_estimate:.12e}")

    print()
    print("Runtime")
    print("-" * 78)
    print(
        f"Mean estimator time           : "
        f"{mean_estimator_time:.6f} s"
    )
    print(
        f"Std estimator time            : "
        f"{std_estimator_time:.6f} s"
    )
    print(
        f"Spectral setup time (once)    : "
        f"{spectral_time_once:.6f} s"
    )

    # --------------------------------------------------------
    # Error statistics
    # --------------------------------------------------------
    results = {
        "seeds": seeds,
        "estimates": estimates,
        "mean_estimate": mean_estimate,
        "std_estimate": std_estimate,
        "median_estimate": median_estimate,
        "q1_estimate": q1_estimate,
        "q3_estimate": q3_estimate,
        "iqr_estimate": iqr_estimate,
        "estimator_times": estimator_times,
        "mean_estimator_time": mean_estimator_time,
        "std_estimator_time": std_estimator_time,
        "spectral_time_once": spectral_time_once,
        "matvecs_per_trial": target_budget,
        "total_matvecs_all_trials": int(np.sum(matvecs)),
    }

    if reference_logdet is not None:

        absolute_errors = np.asarray(
            absolute_errors,
            dtype=float
        )

        relative_errors = np.asarray(
            relative_errors,
            dtype=float
        )

        # Error of the averaged logdet estimate
        error_of_mean = abs(
            mean_estimate - reference_logdet
        )

        relative_error_of_mean = (
            error_of_mean /
            max(abs(reference_logdet), 1e-300)
        )

        # Statistics of individual trial errors
        mean_relative_error = float(
            np.mean(relative_errors)
        )

        std_relative_error = float(
            np.std(relative_errors, ddof=1)
        )

        median_relative_error = float(
            np.median(relative_errors)
        )

        q1_rel = float(
            np.percentile(relative_errors, 25)
        )

        q3_rel = float(
            np.percentile(relative_errors, 75)
        )

        iqr_relative_error = q3_rel - q1_rel

        print()
        print("Accuracy")
        print("-" * 78)

        print(
            f"Reference logdet              : "
            f"{reference_logdet:.12e}"
        )

        print(
            f"Error of mean estimate        : "
            f"{error_of_mean:.12e}"
        )

        print(
            f"Relative error of mean        : "
            f"{relative_error_of_mean:.12e}"
        )

        print()
        print(
            f"Mean individual rel. error    : "
            f"{mean_relative_error:.12e}"
        )

        print(
            f"Std individual rel. error     : "
            f"{std_relative_error:.12e}"
        )

        print(
            f"Median individual rel. error  : "
            f"{median_relative_error:.12e}"
        )

        print(
            f"IQR individual rel. error     : "
            f"{iqr_relative_error:.12e}"
        )

        results.update({
            "reference_logdet": reference_logdet,
            "absolute_errors": absolute_errors,
            "relative_errors": relative_errors,
            "error_of_mean": error_of_mean,
            "relative_error_of_mean": relative_error_of_mean,
            "mean_relative_error": mean_relative_error,
            "std_relative_error": std_relative_error,
            "median_relative_error": median_relative_error,
            "iqr_relative_error": iqr_relative_error,
        })

    print("=" * 78)

    return results