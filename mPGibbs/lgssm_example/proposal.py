import numpy as np
import numba as nb

@nb.njit
def q_rvs(theta, tau):
    eps = np.random.randn(*theta.shape)
    return theta + tau ** 0.5 * eps
