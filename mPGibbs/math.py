import numba as nb
import numpy as np

@nb.njit
def norm_logpdf(x, mu, sig):
    return -np.log(sig) - 0.5 * ((x - mu) / sig) ** 2