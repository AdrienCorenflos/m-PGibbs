from typing import NamedTuple

import matplotlib.pyplot as plt
import numba as nb
import numpy as np
import pandas as pd
import seaborn as sns
import tqdm.auto as tqdm

import mPGibbs.lgssm_example.model as model
from mPGibbs.lgssm_example import proposal
from mPGibbs.math import norm_logpdf


class KalmanState(NamedTuple):
    ell: float
    theta: np.ndarray
    trajectory: np.ndarray


def kf(theta, T, ys):
    """
    Kalman filter for the linear Gaussian state space model.
    """
    *_, prior_gaussian = model.prior(None)
    *_, obs_gaussian = model.observation()
    *_, state_gaussian = model.transition()
    m, P = prior_gaussian(theta)
    H, c, R = obs_gaussian(theta)
    F, b, Q = state_gaussian(theta)

    return _kf(T, ys, m, P, H, c, R, F, b, Q)


@nb.njit
def _kf(T, ys, m, P, H, c, R, F, b, Q):
    """
    Particle filter for the linear Gaussian state space model.
    """

    # Initialize the system
    ms = np.zeros((T,))
    Ps = np.zeros((T,))

    # time 0 update
    S = P * H ** 2 + R
    y_mean = H * m + c
    ell = norm_logpdf(ys[0], y_mean, np.sqrt(S))

    K = P * H / S
    ms[0] = m + K * (ys[0] - H * m - c)
    Ps[0] = P - K * H * P

    for t in range(1, T):
        m = F * ms[t - 1] + b
        P = Ps[t - 1] * F ** 2 + Q

        S = P * H ** 2 + R
        y_mean = H * m + c
        ell += norm_logpdf(ys[t], y_mean, np.sqrt(S))

        K = P * H / S
        ms[t] = m + K * (ys[t] - y_mean)
        Ps[t] = P - K * H * P

    # initialize the chosen trajectory
    out = np.zeros((T,))
    ms = ms[::-1]
    Ps = Ps[::-1]

    out[0] = ms[0] + Ps[0] ** 0.5 * np.random.randn()
    for t in range(1, T):
        m = ms[t]
        P = Ps[t]

        S = P * F ** 2 + Q
        next_x_mean = F * m + b
        K = P * F / S
        m_samp = m + K * (out[t - 1] - next_x_mean)
        P_samp = P - K * F * P
        out[t] = m_samp + P_samp ** 0.5 * np.random.randn()

        m_marg = m + K * (ms[t - 1] - next_x_mean)
        P_marg = P + K ** 2 * (Ps[t - 1] - S)
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


def do_one(K, T, burn, delta):
    theta_rvs, *_ = model.theta_prior()
    ys = np.loadtxt('./simulated_linGauss_T100_varX1_varY.04_rho.9.txt')

    trajectories = np.zeros((K, T))
    parameters = pd.DataFrame(np.zeros((K, 3)), columns=["rho", r"$\sigma_x^2$", r"$\sigma_y^2$"])

    theta_init = theta_rvs()

    traj, ell, ms, Ps = kf(theta_init, T, ys)

    trajectories[0] = traj
    parameters.iloc[0] = theta_init

    state = KalmanState(ell, theta_init, traj)
    accepted = np.zeros((K, T), dtype=bool)
    accepted[0] = True

    pbar = tqdm.trange(1, K, desc="Kalman: pct accepted")
    for k in pbar:
        state, accepted[k] = mh(state, T, ys, delta)
        trajectories[k] = state.trajectory
        parameters.iloc[k] = state.theta
        pbar.set_description(f"Kalman: {np.mean(accepted[:k + 1]):.2%} accepted")

    accepted = accepted[burn:]

    return np.mean(accepted)





if __name__ == "__main__":
    import joblib

    Ts = [100]
    K = 10_000
    burn = K // 10
    delta = 0.15 ** 2

    results = joblib.Parallel(n_jobs=-1)(
        joblib.delayed(do_one)(K, T, burn, delta) for T in Ts
    )

    np.savetxt("kalman_results.txt", results)




# if __name__ == "__main__":
#     # Test the Kalman function
#     np.random.seed(0)
#     T = 100
#     K = 1_000
#     delta = 0.15 ** 2
#     burn = K // 10
#
#     theta_rvs, _ = model.theta_prior()
#     ys = np.loadtxt('./simulated_linGauss_T100_varX1_varY.04_rho.9.txt')
#
#     trajectories = np.zeros((K, T))
#     parameters = pd.DataFrame(np.zeros((K, 3)), columns=["rho", r"$\sigma_x^2$", r"$\sigma_y^2$"])
#
#     theta_init = theta_rvs()
#     traj, ell, ms, Ps = kf(theta_init, T, ys)
#     trajectories[0] = traj
#     parameters.iloc[0] = theta_init
#
#     state = KalmanState(ell, theta_init, traj)
#     accepted = np.zeros((K, T), dtype=bool)
#     accepted[0] = True
#
#     pbar = tqdm.trange(1, K, desc="MH: pct accepted")
#     for k in pbar:
#         state, accepted[k] = mh(state, T, ys, delta)
#         trajectories[k] = state.trajectory
#         parameters.iloc[k] = state.theta
#         pbar.set_description(f"MH: {np.mean(accepted[:k + 1]):.2%} accepted")
#
#     trajectories = trajectories[burn:]
#     parameters = parameters.iloc[burn:]
#
#     g = sns.pairplot(parameters, kind="kde")
#
#     true_theta = np.array([0.9, 1, 0.2 ** 2])
#     for i in range(3):
#         for j in range(3):
#             if i != j:
#                 g.axes[i, j].scatter(
#                     true_theta[j], true_theta[i],
#                     color="k", label="True Value", zorder=10
#                 )
#             else:
#                 g.axes[i, j].vlines(
#                     true_theta[i], 0, 1, transform=g.axes[i, j].get_xaxis_transform(), color="k",
#                 )
#
#     plt.show()
#
#
#     plt.plot(trajectories.mean(axis=0), label="Mean Trajectory", color='red')
#     plt.fill_between(
#         np.arange(T),
#         trajectories.mean(axis=0) - 2 * trajectories.std(axis=0),
#         trajectories.mean(axis=0) + 2 * trajectories.std(axis=0),
#         alpha=0.5,
#         label="Min/Max Trajectory",
#         color='blue'
#     )
#     plt.plot(np.arange(T), ys[:T], label="Observations", color='k', marker='o', markersize=2, linestyle='None')
#     # plt.ylim(-0.75, 1.25)
#     plt.suptitle("Kalman")
#     plt.show()
#
