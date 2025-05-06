from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import tqdm.auto as tqdm
from scipy.stats import norm

import mPGibbs.lgssm_example.model as model
from mPGibbs.lgssm_example import proposal


class KalmanState(NamedTuple):
    ell: float
    theta: np.ndarray
    trajectory: np.ndarray


def kf(theta, T, ys):
    """
    Particle filter for the linear Gaussian state space model.
    """
    # Initialize the model
    *_, prior_gaussian = model.prior(None)
    *_, obs_gaussian = model.observation()
    *_, state_gaussian = model.transition()
    m, P = prior_gaussian(theta)
    H, c, R = obs_gaussian(theta)
    F, b, Q = state_gaussian(theta)

    # Initialize the system
    ms = np.zeros((T,))
    Ps = np.zeros((T,))

    # time 0 update
    S = P * H ** 2 + R
    y_mean = H * m + c
    ell = norm.logpdf(ys[0], loc=y_mean, scale=np.sqrt(S))

    K = P * H / S
    ms[0] = m + K * (ys[0] - H * m - c)
    Ps[0] = P - K * H * P

    for t in range(1, T):
        m = F * ms[t - 1] + b
        P = Ps[t - 1] * F ** 2 + Q

        S = P * H ** 2 + R
        y_mean = H * m + c
        ell += norm.logpdf(ys[t], loc=y_mean, scale=np.sqrt(S))
        K = P * H / S
        ms[t] = m + K * (ys[t] - y_mean)
        Ps[t] = P - K * H * P


    # initialize the chosen trajectory
    out = np.zeros((T,))
    ms = ms[::-1]
    Ps = Ps[::-1]

    out[0] = norm.rvs(loc=ms[0], scale=Ps[0] ** 0.5)
    for t in range(1, T):
        m = ms[t]
        P = Ps[t]

        S = P * F ** 2 + Q
        next_x_mean = F * m + b
        K = P * F / S
        m_samp = m + K * (out[t - 1] - next_x_mean)
        P_samp = P - K * F * P
        out[t] = norm.rvs(loc=m_samp, scale=P_samp ** 0.5)

        m_marg = m + K * (ms[t-1] - next_x_mean)
        P_marg = P + K ** 2 * (Ps[t-1] - S)
        ms[t] = m_marg
        Ps[t] = P_marg
    return out[::-1], ell, ms[::-1], Ps[::-1]


def mh(state, T, ys, delta):
    ell, theta, trajectory = state
    _, theta_logpdf = model.theta_prior()

    # Propose a new theta
    theta_new = proposal.q_rvs(theta, delta)

    # Run the particle filter
    trajectory_new, ell_new, *_ = kf(theta_new, T, ys)

    # Compute the acceptance ratio
    log_alpha = ell_new + theta_logpdf(theta_new) - ell - theta_logpdf(theta)
    alpha = np.exp(log_alpha)
    if np.random.rand() < alpha:
        return KalmanState(ell_new, theta_new, trajectory_new), True
    else:
        return KalmanState(ell, theta, trajectory), False


if __name__ == "__main__":
    # Test the Kalman function
    np.random.seed(0)
    T = 100
    K = 1000
    delta = 0.

    theta_rvs, _ = model.theta_prior()
    theta = np.array([0.9, 0.1, 0.25])
    xs, ys = model.get_data(theta, T)

    trajectories = np.zeros((K, T))

    traj, ell, ms, Ps = kf(theta, T, ys)
    trajectories[0] = traj

    state = KalmanState(ell, theta, traj)
    for k in tqdm.trange(1, K):
        state = mh(state, T, ys, delta)
        trajectories[k] = state.trajectory

    # plt.plot(trajectories.T, label="Trajectories", alpha=0.1)
    # plt.show()

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

    plt.plot(ms, label="Mean Trajectory", color='red')
    plt.fill_between(
        np.arange(T),
        ms - 2 * Ps ** 0.5,
        ms + 2 * Ps ** 0.5,
        alpha=0.5,
        label="Min/Max Trajectory",
        color='blue'
    )
    plt.plot(np.arange(T), ys, label="Observations", color='k', marker='o', markersize=2, linestyle='None')
    plt.ylim(-0.75, 1.25)

    plt.show()