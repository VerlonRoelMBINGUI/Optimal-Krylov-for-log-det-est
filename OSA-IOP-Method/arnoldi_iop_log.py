import numpy as np
import scipy.linalg as la


class ArnoldiIOPLog:
    """
    Arnoldi + length-2 IOP approximation of log(A)v.
    """

    def __init__(self, Aop, tol=1e-6, m_max=100, gamma=None):
        self.Aop = Aop
        self.tol = tol
        self.m_max = m_max
        self.gamma = gamma

    def apply(self, v, add_log_gamma=True):

        n = v.size
        beta0 = la.norm(v)

        if beta0 == 0:
            return np.zeros_like(v)

        v1 = v / beta0

        if self.gamma is not None:
            c = np.log(self.gamma)

            def Atilde(x):
                return self.Aop(x) / self.gamma
        else:
            c = 0.0
            Atilde = self.Aop

        V = np.zeros((n, self.m_max + 1))
        H = np.zeros((self.m_max + 1, self.m_max))

        V[:, 0] = v1

        m = 0
        h_next = np.inf

        while (m < self.m_max) and (h_next >= self.tol):

            w = Atilde(V[:, m])

            if m >= 1:
                for j in range(max(0, m - 1), m + 1):
                    H[j, m] = np.dot(V[:, j], w)
                    w -= H[j, m] * V[:, j]
            else:
                H[m, m] = np.dot(V[:, m], w)
                w -= H[m, m] * V[:, m]

            h_next = la.norm(w)
            H[m + 1, m] = h_next

            if h_next < self.tol:
                break

            V[:, m + 1] = w / h_next
            m += 1

        m_used = self.m_max if m == self.m_max else m + 1

        Hm = H[:m_used, :m_used]
        Vm = V[:, :m_used]

        Lm = la.logm(Hm)

        e1 = np.zeros(m_used)
        e1[0] = 1.0

        y = beta0 * (Vm @ (Lm @ e1))

        if self.gamma is not None and add_log_gamma:
            y += c * v

        return y