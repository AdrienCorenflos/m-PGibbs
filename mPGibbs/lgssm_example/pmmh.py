from typing import NamedTuple

import matplotlib.pyplot as plt
import numba as nb
import numpy as np
import pandas as pd
import seaborn as sns
import tqdm.auto as tqdm

import mPGibbs.lgssm_example.model as model
from mPGibbs.common.resampling import multinomial
from mPGibbs.lgssm_example import proposal


@nb.njit
def logsumexp(a):
    """
    Compute the logsumexp of an array.
    """
    a_max = np.max(a)
    return a_max + np.log(np.sum(np.exp(a - a_max)))


def protected_choice(w):
    if np.all(np.isfinite(w)):
        return np.random.choice(len(w), p=w)
    else:
        return 0



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

    ell = 0.
    xs = np.zeros((T, N))
    xs[0] = prior_rvs(theta)
    log_ws = np.zeros((T, N))
    ancestors = np.zeros((T - 1, N), dtype=np.int_)

    for t in range(1, T):
        log_ws[t - 1] = obs_logpdf(xs[t - 1], ys[t - 1], theta)
        ls = logsumexp(log_ws[t - 1])
        ell += ls

        w = np.exp(log_ws[t - 1] - ls)
        idx = multinomial(w, False)
        ancestors[t - 1] = idx

        xs[t] = state_rvs(xs[t - 1, idx], theta)

    log_ws[T - 1] = obs_logpdf(xs[T - 1], ys[T - 1], theta)
    ls = logsumexp(log_ws[T - 1])
    ell += ls
    w = np.exp(log_ws[T - 1] - ls)

    # initialize the chosen trajectory
    out = np.zeros((T,))

    k = protected_choice(w)
    xs = xs[::-1]
    out[0] = xs[0, k]

    if not backward:
        for t in range(1, T):
            k = ancestors[t - 1, k]
            out[t] = xs[t, k]

    else:
        log_ws = log_ws[::-1]

        for t in range(1, T):
            log_w = log_ws[t]
            log_w += state_logpdf(xs[t], out[t - 1], theta)
            w = np.exp(log_w - logsumexp(log_w))
            k = protected_choice(w)
            out[t] = xs[t, k]
    return out[::-1], ell - T * np.log(N)


def pmmh(state, T, N, ys, delta, backward=False):
    ell, theta, trajectory = state
    _, theta_logpdf = model.theta_prior()

    # Propose a new theta
    theta_new = proposal.q_rvs(theta, delta)
    theta_new_logpdf = theta_logpdf(theta_new)
    if not np.isfinite(theta_new_logpdf):
        return state, False

    # Run the particle filter
    trajectory_new, ell_new = pf(theta_new, T, N, ys, backward)

    # Compute the acceptance ratio
    log_alpha = ell_new + theta_new_logpdf - ell - theta_logpdf(theta)
    alpha = np.exp(log_alpha)
    if np.random.rand() < alpha:
        return PMMHState(ell_new, theta_new, trajectory_new), True
    else:
        return PMMHState(ell, theta, trajectory), False


def do_one(K, T, N, burn, delta):
    theta_rvs, *_ = model.theta_prior()
    ys = np.loadtxt('./simulated_linGauss_T100_varX1_varY.04_rho.9.txt')

    trajectories = np.zeros((K, T))
    parameters = pd.DataFrame(np.zeros((K, 3)), columns=["rho", r"$\sigma_x^2$", r"$\sigma_y^2$"])

    theta_init = theta_rvs()

    traj, ell = pf(theta_init, T, N, ys, backward=True)

    trajectories[0] = traj
    parameters.iloc[0] = theta_init

    state = PMMHState(ell, theta_init, traj)
    accepted = np.zeros((K, T), dtype=bool)
    accepted[0] = True

    pbar = tqdm.trange(1, K, desc="PMMH: pct accepted")
    for k in pbar:
        state, accepted[k] = pmmh(state, T, N, ys, delta, True)
        trajectories[k] = state.trajectory
        parameters.iloc[k] = state.theta
        pbar.set_description(f"PMMH: {np.mean(accepted[:k + 1]):.2%} accepted")


    accepted = accepted[burn:]

    return np.mean(accepted)




if __name__ == "__main__":
    import joblib

    T = 100
    Ns = [16, 32, 64, 128, 256]
    K = 10_000
    burn = K // 10
    delta = 0.15 ** 2

    results = joblib.Parallel(n_jobs=-1)(
        joblib.delayed(do_one)(K, T, N, burn, delta) for N in Ns
    )

    np.savetxt("pmmh_results.txt", results)

#
# if __name__ == "__main__":
#     # Test the PMMH function
#     np.random.seed(0)
#     T = 100
#     N = 15
#     K = 1000
#     burn = K // 10
#     delta = 0.15 ** 2
#
#     theta_rvs, *_ = model.theta_prior()
#     ys = np.loadtxt('./simulated_linGauss_T100_varX1_varY.04_rho.9.txt')
#
#     trajectories = np.zeros((K, T))
#     parameters = pd.DataFrame(np.zeros((K, 3)), columns=["rho", r"$\sigma_x^2$", r"$\sigma_y^2$"])
#
#     theta_init = theta_rvs()
#
#     traj, ell = pf(theta_init, T, N, ys, backward=True)
#
#     trajectories[0] = traj
#     parameters.iloc[0] = theta_init
#
#     state = PMMHState(ell, theta_init, traj)
#     accepted = np.zeros((K, T), dtype=bool)
#     accepted[0] = True
#
#     pbar = tqdm.trange(1, K, desc="PMMH: pct accepted")
#     for k in pbar:
#         state, accepted[k] = pmmh(state, T, N, ys, delta, True)
#         trajectories[k] = state.trajectory
#         parameters.iloc[k] = state.theta
#         pbar.set_description(f"PMMH: {np.mean(accepted[:k + 1]):.2%} accepted")
#
#     trajectories = trajectories[burn:]
#     parameters = parameters.iloc[burn:]
#     #
#     # g = sns.pairplot(parameters, kind="kde")
#     # true_theta = np.array([0.9, 1, 0.2 ** 2])
#     # for i in range(3):
#     #     for j in range(3):
#     #         if i != j:
#     #             g.axes[i, j].scatter(
#     #                 true_theta[j], true_theta[i],
#     #                 color="k", label="True Value", zorder=10
#     #             )
#     #         else:
#     #             g.axes[i, j].vlines(
#     #                 true_theta[i], 0, 1, transform=g.axes[i, j].get_xaxis_transform(), color="k",
#     #             )
#     #
#     # plt.show()
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
#     plt.suptitle("PMMH")
#
#     # plt.ylim(-0.75, 1.25)
#     plt.show()
