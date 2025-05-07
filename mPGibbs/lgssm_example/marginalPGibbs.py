from typing import NamedTuple

import matplotlib.pyplot as plt
import numba as nb
import numpy as np
import pandas as pd
import seaborn as sns
import tqdm.auto as tqdm
from scipy.special import logsumexp as scipy_logsumexp

import mPGibbs.lgssm_example.model as model
from mPGibbs.common.resampling import multinomial
from mPGibbs.lgssm_example import proposal
from mPGibbs.lgssm_example.pmmh import pf


@nb.njit
def logsumexp(a):
    """
    Compute the logsumexp of an array.
    """
    a_max = np.max(a)
    return a_max + np.log(np.sum(np.exp(a - a_max)))


def vectorized_choice(w, axis=1):
    r = np.expand_dims(np.random.rand(w.shape[1 - axis]), axis=axis)
    return (w.cumsum(axis=axis) > r).argmax(axis=axis)


def protected_choice(w):
    if np.all(np.isfinite(w)):
        return np.random.choice(len(w), p=w)
    else:
        return 0


def update_theta_weights(thetas, log_ws, x, x_prev, y, state_logpdf, obs_logpdf):
    log_ws = np.copy(log_ws)
    for i, theta in enumerate(thetas):
        log_ws[i] += state_logpdf(x_prev, x, theta) + obs_logpdf(x, y, theta)
    log_ws -= logsumexp(log_ws)
    return np.exp(log_ws)


def current_transition(transition_rvs, transition_logpdf):
    def rvs(x, thetas, *_):
        return transition_rvs(x, thetas[0])

    def logpdf(x_prev, x, thetas, *_):
        return transition_logpdf(x_prev, x, thetas[0])

    return rvs, logpdf


def current_prior(prior_rvs, prior_logpdf):
    def rvs(thetas, *_):
        return prior_rvs(thetas[0])

    def logpdf(x, thetas, *_):
        return prior_logpdf(x, thetas[0])

    return rvs, logpdf


def proposal_transition(transition_rvs, transition_logpdf):
    def rvs(x, thetas, theta_ws):
        theta_ws_except_0 = theta_ws / (1 - theta_ws[0])[None, :]
        ks = vectorized_choice(theta_ws_except_0[1:], axis=0) + 1
        return transition_rvs(x, thetas[ks, :].T)

    def logpdf(x_prev, x, thetas, theta_ws):
        theta_ws_except_0 = theta_ws / (1 - theta_ws[0])[None, :]
        logpdf = np.zeros((len(thetas) - 1, x_prev.shape[0]))
        for i, theta in enumerate(thetas[1:]):
            logpdf[i] = transition_logpdf(x_prev, x, theta)
        return scipy_logsumexp(logpdf, axis=0, b=theta_ws_except_0[1:])

    return rvs, logpdf


def proposal_prior(prior_rvs, prior_logpdf):
    def rvs(thetas, theta_ws):
        theta_ws_except_0 = theta_ws / (1 - theta_ws[0])[None, :]
        ks = vectorized_choice(theta_ws_except_0[1:], axis=0) + 1
        return prior_rvs(thetas[ks].T)

    def logpdf(x, thetas, theta_ws):
        theta_ws_except_0 = theta_ws / (1 - theta_ws[0])[None, :]
        logpdf = np.zeros((len(thetas) - 1, x.shape[0]))
        for i, theta in enumerate(thetas[1:]):
            logpdf[i] = prior_logpdf(x, theta)
        return scipy_logsumexp(logpdf, axis=0, b=theta_ws_except_0[1:])

    return rvs, logpdf

def mixture_transition(transition_rvs, transition_logpdf):
    def rvs(x, thetas, theta_ws):
        ks = vectorized_choice(theta_ws, 0)
        return transition_rvs(x, thetas[ks].T)

    def logpdf(x_prev, x, thetas, theta_ws):
        logpdf = np.zeros((len(thetas), x_prev.shape[0]))
        for i, theta in enumerate(thetas):
            logpdf[i] = transition_logpdf(x_prev, x, theta)
        return scipy_logsumexp(logpdf, axis=0, b=theta_ws)

    return rvs, logpdf


def mixture_prior(prior_rvs, prior_logpdf):
    def rvs(thetas, theta_ws):
        ks = vectorized_choice(theta_ws, 0)
        return prior_rvs(thetas[ks].T)

    def logpdf(x, thetas, theta_ws):
        logpdf = np.zeros((len(thetas), x.shape[0]))
        for i, theta in enumerate(thetas):
            logpdf[i] = prior_logpdf(x, theta)
        return scipy_logsumexp(logpdf, axis=0, b=theta_ws)

    return rvs, logpdf


def eta_t(transition_logpdf, obs_logpdf):
    def log_weight(x_prev, x, y, thetas, thetas_ws):
        logpdf = np.zeros((len(thetas), x_prev.shape[0]))
        for i, theta in enumerate(thetas):
            logpdf[i] = transition_logpdf(x_prev, x, theta) + obs_logpdf(x, y, theta)
        return scipy_logsumexp(logpdf, axis=0, b=thetas_ws)

    return log_weight


def eta_0(prior_logpdf, obs_logpdf):
    def log_weight(x, y, thetas, thetas_ws):
        logpdf = np.zeros((len(thetas), x.shape[0]))
        for i, theta in enumerate(thetas):
            logpdf[i] = prior_logpdf(x, theta) + obs_logpdf(x, y, theta)
        return scipy_logsumexp(logpdf, axis=0, b=thetas_ws)

    return log_weight


class MPGibbsState(NamedTuple):
    theta: np.ndarray
    trajectory: np.ndarray


def cpf(thetas, trajectory, T, N, M, ys, backward=True, style="proposal"):
    """
    Particle filter for the linear Gaussian state space model.
    """
    # Initialize the model
    _, theta_logpdf = model.theta_prior()
    prior_rvs, prior_logpdf, _ = model.prior(N)
    obs_rvs, obs_logpdf, _ = model.observation()
    state_rvs, state_logpdf, _ = model.transition()

    # initialize the theta weights
    theta_log_ws = np.zeros((T, M, N))
    theta_ws = np.zeros((T, M, N))
    for m, theta in enumerate(thetas):
        theta_log_ws[0, m, :] = theta_logpdf(theta)

    # normalize the theta weights along the M dimension
    theta_log_ws[0] -= scipy_logsumexp(theta_log_ws[0], axis=0, keepdims=True)
    theta_ws[0] = np.exp(theta_log_ws[0])

    # pick the proposal model
    if style == "current":
        proposal_t_rvs, proposal_t_logpdf = current_transition(state_rvs, state_logpdf)
        proposal_0_rvs, proposal_0_logpdf = current_prior(prior_rvs, prior_logpdf)
    elif style == "proposal":
        proposal_t_rvs, proposal_t_logpdf = proposal_transition(state_rvs, state_logpdf)
        proposal_0_rvs, proposal_0_logpdf = proposal_prior(prior_rvs, prior_logpdf)
    elif style == "mixture":
        proposal_t_rvs, proposal_t_logpdf = mixture_transition(state_rvs, state_logpdf)
        proposal_0_rvs, proposal_0_logpdf = mixture_prior(prior_rvs, prior_logpdf)
    else:
        raise ValueError("style must be one of 'current', 'proposal', or 'mixture'")

    # initialize the eta weights
    eta_t_logpdf = eta_t(state_logpdf, obs_logpdf)
    eta_0_logpdf = eta_0(prior_logpdf, obs_logpdf)

    xs = np.zeros((T, N))
    log_ws = np.zeros((T, N))
    ancestors = np.zeros((T - 1, N), dtype=np.int_)

    xs[0] = proposal_0_rvs(thetas, theta_ws[0])
    xs[0, 0] = trajectory[0]
    log_ws[0] = eta_0_logpdf(xs[0], ys[0], thetas, theta_ws[0]) - proposal_0_logpdf(xs[0], thetas, theta_ws[0])
    for m, theta in enumerate(thetas):
        theta_log_ws[0, m] += obs_logpdf(xs[0], ys[0], theta) + prior_logpdf(xs[0], theta)
    theta_log_ws[0] -= scipy_logsumexp(theta_log_ws[0], axis=0, keepdims=True)
    theta_ws[0] = np.exp(theta_log_ws[0])

    for t in range(1, T):
        ls = logsumexp(log_ws[t - 1])
        w = np.exp(log_ws[t - 1] - ls)
        idx = multinomial(w, True)

        ancestors[t - 1] = idx
        theta_log_ws_here = theta_log_ws[t - 1, :, idx].T
        theta_ws_here = theta_ws[t - 1, :, idx].T
        x_t_m_1_here = xs[t - 1, idx]
        xs[t] = proposal_t_rvs(x_t_m_1_here, thetas, theta_ws_here)
        xs[t, 0] = trajectory[t]
        log_ws[t] = eta_t_logpdf(x_t_m_1_here, xs[t], ys[t], thetas, theta_ws_here) - proposal_t_logpdf(x_t_m_1_here, xs[t],
                                                                                                 thetas, theta_ws_here)
        for m, theta in enumerate(thetas):
            theta_log_ws[t, m] = obs_logpdf(xs[t], ys[t], theta) + state_logpdf(x_t_m_1_here, xs[t], theta) + \
                                 theta_log_ws_here[m]
        theta_log_ws[t] -= scipy_logsumexp(theta_log_ws[t], axis=0, keepdims=True)
        theta_ws[t] = np.exp(theta_log_ws[t])

    # initialize the chosen trajectory
    out = np.zeros((T,))
    w = np.exp(log_ws[T - 1] - logsumexp(log_ws[T - 1]))
    w_except_0 = w[1:] / (1 - w[0])
    w_except_0 = np.nan_to_num(w_except_0)
    w_except_0 /= np.sum(w_except_0)
    if ~np.all(np.isfinite(w_except_0)):
        k = 0
    else:
        k = np.random.choice(N - 1, p=w_except_0) + 1
        if not np.random.rand() < w[k] / w[0]:
            k = 0

    xs = xs[::-1]
    out[0] = xs[0, k]

    if not backward:
        ws_theta_k = theta_ws[T - 1, :, k]
        ws_theta_except_0 = ws_theta_k[1:] / (1 - ws_theta_k[0])
        if M == 2:
            k_theta = 1
        else:
            k_theta = np.random.choice(M - 1, p=ws_theta_except_0) + 1
        theta_acc = np.random.rand() < ws_theta_k[k_theta] / ws_theta_k[0]
        if not theta_acc:
            k_theta = 0
        for t in range(1, T):
            k = ancestors[t - 1, k]
            out[t] = xs[t, k]
        return out[::-1], thetas[k_theta], theta_acc
    else:
        theta_ws = theta_ws[::-1]
        log_thetas = np.zeros((M,))
        log_ws = log_ws[::-1]
        ys = np.copy(ys[::-1])
        for t in range(1, T):
            log_w = log_ws[t]
            theta_ws_here = theta_ws[t]
            log_w += eta_t_logpdf(xs[t], out[t-1], ys[t-1], thetas, theta_ws_here)
            w = np.exp(log_w - logsumexp(log_w))
            k = protected_choice(w)
            out[t] = xs[t, k]
            log_thetas += obs_logpdf(out[t-1], ys[t-1], thetas.T) + state_logpdf(out[t], out[t-1], thetas.T)
        log_thetas += obs_logpdf(out[T-1], ys[T-1], thetas.T) + prior_logpdf(out[T-1], thetas.T)
        ws_theta_k = np.exp(log_thetas - logsumexp(log_thetas))
        ws_theta_except_0 = ws_theta_k[1:] / (1 - ws_theta_k[0])
        if M == 2:
            k_theta = 1
        else:
            k_theta = np.random.choice(M - 1, p=ws_theta_except_0) + 1
        theta_acc = np.random.rand() < ws_theta_k[k_theta] / ws_theta_k[0]
        if not theta_acc:
            k_theta = 0
        return out[::-1], thetas[k_theta], theta_acc


def mPGibbs(state, T, N, M, ys, delta, backward=True, style="current"):
    theta, trajectory = state
    _, theta_logpdf = model.theta_prior()
    thetas = np.zeros((M, len(theta)))
    # Propose a new u
    u = proposal.q_rvs(theta, 0.5 * delta)
    # Propose a new theta
    thetas[0] = theta
    for m in range(1, M):
        thetas[m] = proposal.q_rvs(u, 0.5 * delta)

    theta_logpdfs = np.zeros((M,))
    flags = np.ones((M,), dtype=bool)
    for m, theta in enumerate(thetas):
        theta_logpdfs[m] = theta_logpdf(theta)

        if not np.isfinite(theta_logpdfs[m]):
            flags[m] = False

    # remove the non-finite thetas
    thetas = thetas[flags]
    if len(thetas) == 1:
        return state, False

    # Run the particle filter
    trajectory, theta, theta_acc = cpf(thetas, trajectory, T, N, len(thetas), ys, backward=backward, style=style)

    return MPGibbsState(theta, trajectory), theta_acc


def do_one(K, T, N, burn, delta, style):
    theta_rvs, *_ = model.theta_prior()
    ys = np.loadtxt('./simulated_linGauss_T100_varX1_varY.04_rho.9.txt')

    trajectories = np.zeros((K, T))
    parameters = pd.DataFrame(np.zeros((K, 3)), columns=["rho", r"$\sigma_x^2$", r"$\sigma_y^2$"])

    theta_init = theta_rvs()

    traj, ell = pf(theta_init, T, N, ys, backward=True)

    trajectories[0] = traj
    parameters.iloc[0] = theta_init

    state = MPGibbsState(theta_init, traj)
    accepted = np.zeros((K, T), dtype=bool)
    accepted[0] = True

    pbar = tqdm.trange(1, K, desc="Gibbs: pct accepted")
    for k in pbar:
        state, accepted[k] = mPGibbs(state, T, N, 2, ys, delta, backward=True, style=style)
        trajectories[k] = state.trajectory
        parameters.iloc[k] = state.theta
        pbar.set_description(f"Gibbs: {np.mean(accepted[:k + 1]):.2%} accepted")

    accepted = accepted[burn:]

    return np.mean(accepted)


if __name__ == "__main__":
    import joblib

    T = 100
    Ns = [16, 32, 64, 128, 256]
    style = "mixture"
    K = 10_000
    burn = K // 10
    delta = 0.15 ** 2

    results = joblib.Parallel(n_jobs=-1)(
        joblib.delayed(do_one)(K, T, N, burn, delta, style) for N in Ns
    )

    np.savetxt(f"pgibbs_results_{style}.txt", results)

