import numpy as np

from intervals import get_focal_interval
from arnoldi_iop_log import ArnoldiIOPLog
from hutchpp import hutchpp_trace


class LogDetEstimator:
    """
    High-level log-det estimator:
    log(det(Q)) via Arnoldi + Hutch++
    """

    def __init__(self, Q, m=21, krylov_dim=30, tol=1e-6):

        self.Q = Q
        self.n = Q.shape[0]

        self.m = m
        self.krylov_dim = krylov_dim
        self.tol = tol

        self.alpha, self.beta = get_focal_interval(Q)
        self.gamma = np.sqrt(self.alpha * self.beta)

        self.c_psd = np.log(self.gamma / self.alpha)

        self.arnoldi = ArnoldiIOPLog(
            Aop=lambda x: Q @ x,
            tol=tol,
            m_max=krylov_dim,
            gamma=self.gamma,
        )

    def estimate(self, rng=42):

        def Ftilde(x):
            return self.arnoldi.apply(x, add_log_gamma=False)

        def Gop(x):
            return Ftilde(x) + self.c_psd * x

        est, info = hutchpp_trace(Gop, self.n, self.m, rng=rng)

        logdet = est + self.n * (np.log(self.gamma) - self.c_psd)

        return float(logdet), info