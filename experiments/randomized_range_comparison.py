import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

Path("results").mkdir(exist_ok=True)


# ============================================================
# PARAMETERS
# ============================================================

seed = 0
rng = np.random.default_rng(seed)

n = 400
k = 10
p = 12
s = k + p

gamma = 1.0
trials = 400


# ============================================================
# NUMERICALLY SAFE ORTHONORMAL BASIS
# ============================================================

def orth_basis(Y, rtol=None):
    """
    Compute an orthonormal basis for range(Y).

    Uses SVD instead of plain QR so that numerically zero
    directions are removed. This is useful when J-gamma*I
    is nearly rank deficient.
    """

    U, sigma, _ = np.linalg.svd(
        Y,
        full_matrices=False
    )

    if sigma.size == 0:
        return np.zeros((Y.shape[0], 0))

    if rtol is None:
        rtol = (
            max(Y.shape)
            * np.finfo(float).eps
        )

    rank = np.sum(
        sigma > rtol * sigma[0]
    )

    return U[:, :rank]


# ============================================================
# ONE EXPERIMENT
# ============================================================

def test(
    lam,
    label,
    trials=400,
    gamma=1.0
):
    """
    Compare the three randomized spaces

        range(J S),
        range((J-gamma I) S),
        range(log(J) S)

    for approximating

        E = log(J).

    The quantity measured is

        ||(I-P)E||_F^2.

    The experiment is carried out directly in the eigenbasis
    of J. For Gaussian sketches this is equivalent to using
    a random orthogonal eigenvector matrix because the
    Gaussian distribution is rotationally invariant.
    """

    lam = np.asarray(
        lam,
        dtype=float
    )

    if np.any(lam <= 0):
        raise ValueError(
            "All eigenvalues must be positive."
        )

    # --------------------------------------------------------
    # Sort eigenvalues in decreasing order
    # --------------------------------------------------------

    lam = np.sort(lam)[::-1]

    L = np.log(lam)

    # Leading and trailing spectral blocks
    l1 = lam[:k]
    l2 = lam[k:]

    L1 = L[:k]
    L2 = L[k:]


    # --------------------------------------------------------
    # Check theorem condition
    # --------------------------------------------------------

    if np.any(
        np.isclose(
            l1,
            gamma
        )
    ):
        raise ValueError(
            "Lambda_1 - gamma I must be invertible."
        )


    # ========================================================
    # THREE RANGE CONSTRUCTIONS
    # ========================================================

    multipliers = {

        # Y = J S
        "JS": lam,

        # Y = (J - gamma I) S
        "(J-I)S": lam - gamma,

        # Y = log(J) S
        "logJ S": L
    }


    # Store all Monte Carlo residuals
    residual_samples = {
        "JS": np.empty(trials),
        "(J-I)S": np.empty(trials),
        "logJ S": np.empty(trials),
    }


    # --------------------------------------------------------
    # ||E||_F^2
    #
    # Since E = diag(log(lambda_i)) in the eigenbasis:
    #
    # ||E||_F^2 = sum_i log(lambda_i)^2
    # --------------------------------------------------------

    E_sq = np.sum(
        L**2
    )


    # ========================================================
    # MONTE CARLO LOOP
    # ========================================================

    for t in range(trials):

        # Use the SAME Gaussian sketch for all three methods.
        #
        # This gives a paired comparison and reduces
        # Monte Carlo noise.

        S = rng.standard_normal(
            (n, s)
        )


        for name, d in multipliers.items():

            # ------------------------------------------------
            # Construct randomized range
            #
            # In eigenbasis:
            #
            # J S              -> diag(lambda) S
            #
            # (J-gamma I) S    -> diag(lambda-gamma) S
            #
            # log(J) S         -> diag(log(lambda)) S
            # ------------------------------------------------

            Y = (
                d[:, None]
                * S
            )


            # A = orth(Y)

            A = orth_basis(Y)


            # ------------------------------------------------
            # Compute
            #
            # ||(I-P)E||_F^2
            #
            # where P = A A^T.
            #
            # Use identity:
            #
            # ||(I-P)E||_F^2
            # =
            # ||E||_F^2
            # -
            # ||A^T E||_F^2.
            #
            # Since E = diag(L),
            #
            # ||A^T E||_F^2
            # =
            # sum_{i,j} (A_ij L_i)^2.
            # ------------------------------------------------

            captured = np.sum(
                (
                    A
                    * L[:, None]
                )**2
            )

            residual = (
                E_sq
                - captured
            )

            # Avoid tiny negative values from roundoff
            residual_samples[name][t] = max(
                residual,
                0.0
            )


    # ========================================================
    # EMPIRICAL EXPECTATIONS
    # ========================================================

    empirical = {
        name:
        float(
            np.mean(values)
        )
        for name, values
        in residual_samples.items()
    }


    std = {
        name:
        float(
            np.std(
                values,
                ddof=1
            )
        )
        for name, values
        in residual_samples.items()
    }


    # ========================================================
    # THEORETICAL BOUNDS
    # ========================================================

    # Optimal rank-k spectral tail
    #
    # ||L2||_F^2
    #
    tail = np.sum(
        L2**2
    )


    # --------------------------------------------------------
    # Bound for range(J S)
    # --------------------------------------------------------

    b_JS = (
        tail
        +
        np.sum(
            l2**2
        )
        *
        np.sum(
            (
                L1 / l1
            )**2
        )
        /
        (p - 1)
    )


    # --------------------------------------------------------
    # Theorem 2.2 bound for
    #
    # range((J-gamma I) S)
    # --------------------------------------------------------

    b_shift = (
        tail
        +
        np.sum(
            (
                l2
                - gamma
            )**2
        )
        *
        np.sum(
            (
                L1
                /
                (
                    l1
                    - gamma
                )
            )**2
        )
        /
        (p - 1)
    )


    # --------------------------------------------------------
    # Standard randomized range finder bound for
    #
    # range(log(J) S)
    # --------------------------------------------------------

    b_hmt = (
        1.0
        +
        k / (p - 1)
    ) * tail


    return {

        "Spectrum":
            label,

        "Empirical JS":
            empirical["JS"],

        "Bound JS":
            b_JS,

        "Empirical (J-I)S":
            empirical["(J-I)S"],

        "Bound shift":
            b_shift,

        "Empirical logJ S":
            empirical["logJ S"],

        "HMT(logJ)":
            b_hmt,

        "Optimal tail":
            tail,

        "Std JS":
            std["JS"],

        "Std (J-I)S":
            std["(J-I)S"],

        "Std logJ S":
            std["logJ S"],
    }


# ============================================================
# SYNTHETIC SPECTRA
# ============================================================

spectra = []


# ------------------------------------------------------------
# 1. Fast exponential decay toward 1
# ------------------------------------------------------------

lam = (
    1.0
    +
    1e3
    * np.exp(
        -np.arange(n) / 8.0
    )
)

spectra.append(
    (
        lam,
        r"Exp., $\lambda_{\min}=1$"
    )
)


# ------------------------------------------------------------
# 2. Algebraic decay toward 1
# ------------------------------------------------------------

lam = (
    1.0
    +
    1e3
    /
    (
        1.0
        + np.arange(n)
    )**1.5
)

spectra.append(
    (
        lam,
        r"Algebraic, $\lambda_{\min}=1$"
    )
)


# ------------------------------------------------------------
# 3. Exponential decay with lambda_min > 1
# ------------------------------------------------------------

lam = (
    np.e
    *
    (
        1.0
        +
        1e3
        * np.exp(
            -np.arange(n) / 8.0
        )
    )
)

spectra.append(
    (
        lam,
        r"Exp., $\lambda_{\min}=e$"
    )
)


# ------------------------------------------------------------
# 4. Flat spectrum
# ------------------------------------------------------------

lam = np.linspace(
    1.0,
    50.0,
    n
)

spectra.append(
    (
        lam,
        "Flat"
    )
)


# ------------------------------------------------------------
# 5. Tail clustered close to 1
# ------------------------------------------------------------

lam = np.concatenate(
    [

        1.0
        +
        np.linspace(
            50.0,
            5.0,
            k + 5
        ),

        1.0
        +
        0.02
        * rng.random(
            n - k - 5
        )
    ]
)

spectra.append(
    (
        lam,
        "Tail near 1"
    )
)


# ------------------------------------------------------------
# 6. Identity + approximately rank-8
# ------------------------------------------------------------

lam = np.concatenate(
    [

        1.0
        +
        np.logspace(
            2,
            0,
            8
        ),

        1.0
        +
        1e-3
        * rng.random(
            n - 8
        )
    ]
)

spectra.append(
    (
        lam,
        "Identity + rank-8"
    )
)


# ============================================================
# RUN ALL EXPERIMENTS
# ============================================================

results = []

for lam, label in spectra:

    result = test(
        lam,
        label,
        trials=trials,
        gamma=gamma
    )

    results.append(result)


df = pd.DataFrame(
    results
)


# ============================================================
# DISPLAY NUMERICAL RESULTS
# ============================================================

columns = [

    "Spectrum",

    "Empirical JS",
    "Bound JS",

    "Empirical (J-I)S",
    "Bound shift",

    "Empirical logJ S",
    "HMT(logJ)",

    "Optimal tail",
]


print()
print("=" * 150)

print(
    "Randomized range comparison"
)

print("=" * 150)

print(
    df[
        columns
    ].to_string(
        index=False
    )
)

print("=" * 150)


# ============================================================
# PLOT
# ============================================================

labels = df[
    "Spectrum"
].tolist()

x = np.arange(
    len(labels)
)

width = 0.11


fig, ax = plt.subplots(
    figsize=(14, 7)
)


# ------------------------------------------------------------
# JS
# ------------------------------------------------------------

ax.bar(
    x - 3 * width,
    df["Empirical JS"],
    width,
    label=r"Empirical $JS$"
)

ax.bar(
    x - 2 * width,
    df["Bound JS"],
    width,
    label=r"Bound $JS$"
)


# ------------------------------------------------------------
# (J-I)S
# ------------------------------------------------------------

ax.bar(
    x - width,
    df["Empirical (J-I)S"],
    width,
    label=r"Empirical $(J-I)S$"
)

ax.bar(
    x,
    df["Bound shift"],
    width,
    label=r"Bound $(J-I)S$"
)


# ------------------------------------------------------------
# log(J)S
# ------------------------------------------------------------

ax.bar(
    x + width,
    df["Empirical logJ S"],
    width,
    label=r"Empirical $\log(J)S$"
)

ax.bar(
    x + 2 * width,
    df["HMT(logJ)"],
    width,
    label=r"HMT bound for $\log(J)S$"
)


# ------------------------------------------------------------
# Optimal tail
# ------------------------------------------------------------

ax.bar(
    x + 3 * width,
    df["Optimal tail"],
    width,
    label=r"Optimal tail $\|L_2\|_F^2$"
)


# ============================================================
# PLOT FORMATTING
# ============================================================

ax.set_yscale(
    "log"
)

ax.set_ylabel(
    r"Squared Frobenius residual"
)

ax.set_title(
    rf"Randomized range comparison: "
    rf"$n={n}$, "
    rf"$k={k}$, "
    rf"$p={p}$, "
    rf"$\gamma={gamma}$, "
    rf"{trials} trials"
)

ax.set_xticks(
    x
)

ax.set_xticklabels(
    labels,
    rotation=25,
    ha="right"
)

ax.legend(
    ncol=2
)

ax.legend(
    ncol=2
)

ax.grid(
    axis="y",
    alpha=0.25
)

fig.tight_layout()

# Save publication-quality figure
fig.savefig(
    "results/randomized_range_comparison.pdf",
    bbox_inches="tight"
)

fig.savefig(
    "results/randomized_range_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()