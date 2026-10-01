"""Run DSLQ on the SuiteSparse ecology2 matrix with an exact 1800-matvec budget."""

from pathlib import Path
import sys

# Allow running this file directly from the repository root.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.dslq import run_ecology2


if __name__ == "__main__":
    estimate, info = run_ecology2(
        file_path="data/ecology2.mat",
        target_budget=1800,
        mvec=30,
        gamma=1,
        sketch="gaussian",  # change to "rademacher" if desired
        seed=42,
        reference_logdet=None,
    )
