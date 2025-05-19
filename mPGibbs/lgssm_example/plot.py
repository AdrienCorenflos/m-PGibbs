import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

kalman_rate_barker, kalman_rate_mh = np.loadtxt("kalman_results.txt")
Ns = [16, 32, 64, 128, 256, 512]
pmmh_res = np.loadtxt("pmmh_results.txt")
current_gibbs = np.loadtxt("pgibbs_results_current.txt")
proposal_gibbs = np.loadtxt("pgibbs_results_proposal.txt")
mixture_gibbs = np.loadtxt("pgibbs_results_mixture.txt")

df = pd.DataFrame(columns=["PMMH","m-PGibbs"],
                  index=Ns)

df["PMMH"] = pmmh_res
df["m-PGibbs"] = mixture_gibbs
df.index.name = "N"
df.columns.name = "Method"

fig, ax = plt.subplots(figsize=(10, 5))
df.plot(ax=ax, marker="o", markersize=8, fontsize=18)
ax.axhline(kalman_rate_mh, color="k", linestyle="--", label="Ideal MH sampler")
ax.axhline(kalman_rate_barker, color="k", linestyle=":", label="Ideal Barker sampler")
ax.set_xlabel("N", fontsize=18)
ax.set_title("Acceptance Rate vs N", fontsize=20)
ax.legend(fontsize=18)
plt.xticks(fontsize=18)
plt.xscale("log", base=2)
plt.yticks(fontsize=18)
plt.ylim(0, 0.3)
# make y-axis a percentage
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0%}'.format(x)))
ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())

plt.grid()
plt.tight_layout()
plt.savefig("acceptance_rate.pdf")