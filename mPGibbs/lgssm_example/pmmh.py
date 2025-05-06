import numpy as np
from typing import NamedTuple

import tqdm.auto as tqdm

from mPGibbs.common.resampling import multinomial
import mPGibbs.lgssm_example.model as model
import matplotlib.pyplot as plt
from mPGibbs.lgssm_example import proposal
import numba as nb

@nb.njit
def logsumexp(a):
    """
    Compute the logsumexp of an array.
    """
    a_max = np.max(a)
    return a_max + np.log(np.sum(np.exp(a - a_max)))


class PMMHState(NamedTuple):
    ell: float
    theta: np.ndarray
    trajectory: np.ndarray


def pf(theta, T, N, ys, backward=False):
    """
    Particle filter for the linear Gaussian state space model.
    """
    # Initialize the model
    prior_rvs, *_ = model.prior(N)
    obs_rvs, obs_logpdf, _ = model.observation()
    state_rvs, state_logpdf, _ = model.transition()

    # Initialize the system
    ell = 0.
    xs = np.zeros((T, N))
    xs[0] = prior_rvs(theta)
    log_ws = np.zeros((T, N))
    ancestors = np.zeros((T-1, N), dtype=int)

    for t in range(1, T):
        log_ws[t-1] = obs_logpdf(xs[t-1], ys[t-1], theta)
        ls = logsumexp(log_ws[t-1])
        ell += ls

        w = np.exp(log_ws[t-1] - ls)
        idx = multinomial(w, False)
        ancestors[t-1] = idx

        xs[t] = state_rvs(xs[t - 1, idx], theta)

    log_ws[T-1] = obs_logpdf(xs[T-1], ys[T-1], theta)
    ls = logsumexp(log_ws[T-1])
    ell += ls
    w = np.exp(log_ws[T-1] - ls)

    # initialize the chosen trajectory
    out = np.zeros((T,))

    k = np.random.choice(N, p=w)
    xs = xs[::-1]
    out[0] = xs[0, k]

    if not backward:
        for t in range(1, T):
            k = ancestors[t-1, k]
            out[t] = xs[t, k]

    else:
        log_ws = log_ws[::-1]

        for t in range(1, T):
            log_w = log_ws[t]
            log_w += state_logpdf(xs[t], out[t-1], theta)
            w = np.exp(log_w - logsumexp(log_w))
            k = np.random.choice(N, p=w)
            out[t] = xs[t, k]
    return out[::-1], ell - T * np.log(N)



def pmmh(state, T, N, ys, delta, backward=False):
    ell, theta, trajectory = state
    _, theta_logpdf = model.theta_prior()

    # Propose a new theta
    theta_new = proposal.q_rvs(theta, delta)

    # Run the particle filter
    trajectory_new, ell_new = pf(theta_new, T, N, ys, backward)

    # Compute the acceptance ratio
    log_alpha = ell_new + theta_logpdf(theta_new) - ell - theta_logpdf(theta)
    alpha = np.exp(log_alpha)
    if np.random.rand() < alpha:
        return PMMHState(ell_new, theta_new, trajectory_new), True
    else:
        return PMMHState(ell, theta, trajectory), False


if __name__ == "__main__":
    # Test the PMMH function
    np.random.seed(0)
    T = 100
    N = 100
    K = 1000
    burn = 100
    delta = 0.
    import cProfile

    theta_rvs, *_ = model.theta_prior()
    theta = np.array([0.9, 0.1, 0.25])
    xs, ys = model.get_data(theta, T)

    trajectories = np.zeros((K, T))
    accepted = np.zeros((K, T), dtype=bool)


    traj, ell = pf(theta, T, N, ys, True)


    trajectories[0] = traj
    accepted[0] = True
    state = PMMHState(ell, theta, traj)

    pbar = tqdm.trange(1, K, desc="PMMH: pct accepted")
    for k in pbar:
        state, accepted[k] = pmmh(state, T, N, ys, delta, True)
        trajectories[k] = state.trajectory
        pbar.set_description(f"PMMH: {np.mean(accepted[:k+1]):.2%} accepted")


    trajectories = trajectories[burn:]

    plt.plot(trajectories.mean(axis=0), label="Mean Trajectory", color='red')
    plt.fill_between(
        np.arange(T),
        trajectories.mean(axis=0) - 2 * trajectories.std(axis=0),
        trajectories.mean(axis=0) + 2 * trajectories.std(axis=0),
        alpha=0.5,
        label="Min/Max Trajectory",
        color='blue'
    )
    plt.plot(np.arange(T), ys, label="Observations", color='k', marker='o', markersize=2, linestyle='None')

    plt.ylim(-0.75, 1.25)
    plt.show()







