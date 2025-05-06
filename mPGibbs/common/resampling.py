import numba as nb
import numpy as np

@nb.njit
def multinomial(weights, conditional=False):
    N = len(weights)
    cumsum = np.cumsum(weights)
    cumsum /= cumsum[-1]

    us = uniform_spacings(N)
    idx = np.searchsorted(cumsum, us)
    idx = np.clip(idx, 0, N - 1)
    idx = np.random.permutation(idx)
    if conditional:
        idx[0] = 0
    return idx


@nb.njit
def uniform_spacings(N):
    z = np.cumsum(-np.log(np.random.rand(N + 1)))
    return z[:-1] / z[-1]
