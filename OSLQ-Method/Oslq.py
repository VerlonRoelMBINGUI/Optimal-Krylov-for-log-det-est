import numpy as np

from lanczos import lanczos_tridiag, quadratic_form_log_psd
from utils import gaussian_matrix, rademacher, gershgorin_interval
from operators import SymmetricOperator


class SLQLogDetEstimator:
    """
    Stochastic Lanczos + Hutch++ logdet estimator.
    """

    def __init__(self, Q):
        self.op_wrapper = SymmetricOperator(Q)
        self.Aop = self.op_wrapper.op
        self.n = self.op_wrapper.n

    def randomized_range(self, s, q_power, rng):
        n = self.n
        G = gaussian_matrix(n, s, rng)

        Y = np.column_stack([
            np.asarray(self.Aop.matvec(G[:, j]), dtype=float)
            for j in range(s)
        ])

        for _ in range(q_power):
            Z = np.column_stack([
                np.asarray(self.Aop.matvec(Y[:, j]), dtype=float)
                for j in range(s)
            ])
            Y = np.column_stack([
                np.asarray(self.Aop.matvec(Z[:, j]), dtype=float)
                for j in range(s)
            ])

        Q, _ = np.linalg.qr(Y, mode="reduced")
        return Q[:, :s]

    def estimate(
        self,
        m=40,
        n_residual_vecs=50,
        s=20,
        q_power=1,
        rng=None,
        return_breakdown=False,
    ):
        if rng is None:
            rng = np.random.default_rng()

        Q = self.op_wrapper.Q_raw

        _, beta = gershgorin_interval(Q)
        alpha = beta

        basis = self.randomized_range(s, q_power, rng)
        s_eff = basis.shape[1]

        subspace = 0.0
        for i in range(s_eff):
            subspace += quadratic_form_log_psd(
                self.Aop, basis[:, i], m, alpha
            )

        residual = 0.0

        for _ in range(n_residual_vecs):
            z = rademacher(self.n, rng)

            z_perp = z - basis @ (basis.T @ z)
            norm2 = float(np.dot(z_perp, z_perp))

            if norm2 < 1e-20:
                continue

            v = z_perp / np.sqrt(norm2)

            residual += norm2 * quadratic_form_log_psd(
                self.Aop, v, m, alpha
            )

        residual /= max(n_residual_vecs, 1)

        trace_term = subspace + residual
        logdet = self.n * np.log(alpha) - trace_term

        if return_breakdown:
            return logdet, {
                "alpha": alpha,
                "subspace": subspace,
                "residual": residual,
            }

        return logdet