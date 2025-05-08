import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

kalman_rate = np.loadtxt("kalman_results.txt")
Ns = [16, 32, 64, 128, 256, 512]
pmmh_res = np.loadtxt("pmmh_results.txt")
current_gibbs = np.loadtxt("pgibbs_results_current.txt")
proposal_gibbs = np.loadtxt("pgibbs_results_proposal.txt")
mixture_gibbs = np.loadtxt("pgibbs_results_mixture.txt")

df = pd.DataFrame(columns=["PMMH", "m-PGibbs (v1)", "m-PGibbs (v2)", "m-PGibbs (v3)"],
                  index=Ns)

df["PMMH"] = pmmh_res
df["m-PGibbs (v1)"] = current_gibbs
df["m-PGibbs (v2)"] = proposal_gibbs
df["m-PGibbs (v3)"] = mixture_gibbs
df.index.name = "N"
df.columns.name = "Method"

fig, ax = plt.subplots(figsize=(8, 5))
df.plot(ax=ax, marker="o", markersize=8, fontsize=14)
ax.axhline(kalman_rate, color="k", linestyle="--", label="Ideal sampler")
ax.set_xlabel("N", fontsize=16)
ax.set_title("Acceptance Rate vs N", fontsize=16)
ax.legend(fontsize=14)
plt.xticks(fontsize=14)
plt.xscale("log", base=2)
plt.yticks(fontsize=14)
plt.ylim(0, 0.3)
# make y-axis a percentage
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0%}'.format(x)))
plt.grid()
plt.tight_layout()
plt.savefig("acceptance_rate.pdf")