"""Monte Carlo validation of the residual upper bound in Theorem 2.2."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


SEED = 0
N = 400
K = 10
P = 12
S = K + P
GAMMA = 1.0
TRIALS = 400


def shifted_range_experiment(lam, label, rng, trials=TRIALS, gamma=GAMMA):
    """Compare E[||(I-P)log(J)||_F^2] with the Theorem 2.2 upper bound."""
    lam = np.asarray(lam, dtype=float)

    if np.any(lam <= 0):
        raise ValueError("All eigenvalues must be positive.")

    lam = np.sort(lam)[::-1]
    lam1, lam2 = lam[:K], lam[K:]

    log_lam = np.log(lam)
    L1, L2 = log_lam[:K], log_lam[K:]

    if np.any(np.isclose(lam1, gamma)):
        raise ValueError(
            "Lambda_1 - gamma I must be invertible; "
            f"some leading eigenvalue is too close to gamma={gamma}."
        )

    optimal_tail = np.sum(L2**2)
    penalty = (
        np.sum((lam2 - gamma) ** 2)
        / (P - 1)
        * np.sum((L1 / (lam1 - gamma)) ** 2)
    )
    theorem_bound = optimal_tail + penalty

    residuals = np.empty(trials, dtype=float)
    log_j_fro_sq = np.sum(log_lam**2)

    for t in range(trials):
        sketch = rng.standard_normal((N, S))
        Y = (lam - gamma)[:, None] * sketch
        basis, _ = np.linalg.qr(Y, mode="reduced")

        captured = np.sum((basis * log_lam[:, None]) ** 2)
        residuals[t] = max(log_j_fro_sq - captured, 0.0)

    empirical_mean = float(np.mean(residuals))

    return {
        "Spectrum": label,
        "Optimal tail": optimal_tail,
        "Empirical mean": empirical_mean,
        "Empirical std": float(np.std(residuals, ddof=1)),
        "Theorem bound": theorem_bound,
        "Penalty": penalty,
        "Bound / empirical": theorem_bound / empirical_mean,
    }


def build_spectra(rng):
    """Construct the six synthetic spectra used in the validation experiment."""
    return [
        (1.0 + 1e3 * np.exp(-np.arange(N) / 8.0), "Exponential decay"),
        (1.0 + 1e3 / (1.0 + np.arange(N)) ** 1.5, "Algebraic decay"),
        (
            np.e * (1.0 + 1e3 * np.exp(-np.arange(N) / 8.0)),
            r"Exponential, $\lambda_{\min}>1$",
        ),
        (np.linspace(1.0, 50.0, N), "Flat spectrum"),
        (
            np.concatenate(
                [
                    1.0 + np.linspace(50.0, 5.0, K + 5),
                    1.0 + 0.02 * rng.random(N - K - 5),
                ]
            ),
            "Tail near 1",
        ),
        (
            np.concatenate(
                [
                    1.0 + np.logspace(2, 0, 8),
                    1.0 + 1e-3 * rng.random(N - 8),
                ]
            ),
            "Identity + rank-8",
        ),
    ]


def main():
    rng = np.random.default_rng(SEED)
    spectra = build_spectra(rng)

    results = [
        shifted_range_experiment(lam, label, rng=rng)
        for lam, label in spectra
    ]
    df = pd.DataFrame(results)

    print("\nResidual-bound validation")
    print("=" * 100)
    print(
        df[
            [
                "Spectrum",
                "Optimal tail",
                "Empirical mean",
                "Empirical std",
                "Theorem bound",
                "Bound / empirical",
            ]
        ].to_string(index=False)
    )

    labels = df["Spectrum"].tolist()
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, df["Optimal tail"], width, label=r"Rank-$k$ spectral tail $\|L_2\|_F^2$")
    ax.bar(x, df["Empirical mean"], width, label=r"Monte Carlo mean $\|(I-P)E\|_F^2$")
    ax.bar(x + width, df["Theorem bound"], width, label="Theorem 2.2 upper bound")

    ax.set_yscale("log")
    ax.set_ylabel("Squared Frobenius residual")
    ax.set_title(
        rf"Validation of the residual bound: $Y=(J-\gamma I)S$, "
        rf"$\gamma={GAMMA}$, $k={K}$, $p={P}$, trials={TRIALS}"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    fig.savefig(out_dir / "theorem_2_2_residual_bound_validation.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "theorem_2_2_residual_bound_validation.png", dpi=300, bbox_inches="tight")
    df.to_csv(out_dir / "theorem_2_2_residual_bound_validation.csv", index=False)
    plt.show()


if __name__ == "__main__":
    main()
