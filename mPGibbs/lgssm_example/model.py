import numpy as np
from scipy.stats import invgamma, uniform, norm


def transition():
    def rvs(x, theta):
        rho, sig_x, _ = theta
        eps = np.random.randn(*x.shape)
        return rho * x + sig_x * eps

    def logpdf(x_prev, x, theta):
        rho, sig_x, _ = theta
        return norm.logpdf(x, rho * x_prev, sig_x)

    def gaussian(theta):
        rho, sig_x, _ = theta
        F = rho
        b = 0.
        Q = sig_x ** 2
        return F, b, Q

    return rvs, logpdf, gaussian


def theta_prior():
    fro_uniform = uniform(loc=-1, scale=2)
    fro_invgamma = invgamma(a=2, scale=2)

    def rvs():
        rho = fro_uniform.rvs()
        var_x = fro_invgamma.rvs()
        var_y = fro_invgamma.rvs()
        return np.array([rho, np.sqrt(var_x), np.sqrt(var_y)])

    def logpdf(theta):
        rho, sig_x, sig_y = theta
        logpdf_rho = fro_uniform.logpdf(rho)
        logpdf_sig_x = fro_invgamma.logpdf(sig_x ** 2)
        logpdf_sig_y = fro_invgamma.logpdf(sig_y ** 2)
        return logpdf_rho + logpdf_sig_x + logpdf_sig_y

    return rvs, logpdf


def prior(N):
    def rvs(theta):
        rho, sig_x, sig_y = theta
        var_x = sig_x ** 2 / (1 - rho ** 2)
        return var_x ** 0.5 * np.random.randn(N)

    def logpdf(x, theta):
        rho, sig_x, sig_y = theta
        var_x = sig_x ** 2 / (1 - rho ** 2)
        z = x / var_x ** 0.5
        return -0.5 * z ** 2

    def gaussian(theta):
        rho, sig_x, sig_y = theta
        m = 0.
        P = sig_x ** 2 / (1 - rho ** 2)
        return m, P

    return rvs, logpdf, gaussian


def observation():
    def rvs(x, theta):
        rho, sig_x, sig_y = theta
        eps = np.random.randn(*x.shape)
        return x + sig_y * eps

    def logpdf(x, y, theta):
        rho, sig_x, sig_y = theta
        return norm.logpdf(y, x, sig_y)

    def gaussian(theta):
        rho, sig_x, sig_y = theta
        H = 1.
        c = 0.
        R = sig_y ** 2
        return H, c, R

    return rvs, logpdf, gaussian


def get_data(theta, T):
    xs = np.zeros(T)
    ys = np.zeros(T)
    prior_rvs, *_ = prior(1)
    obs_rvs, *_ = observation()
    state_rvs, *_ = transition()

    xs[0] = prior_rvs(theta)[0]
    ys[0] = obs_rvs(xs[0], theta)
    for t in range(1, T):
        xs[t] = state_rvs(xs[t - 1], theta)
        ys[t] = obs_rvs(xs[t], theta)
    return xs, ys
