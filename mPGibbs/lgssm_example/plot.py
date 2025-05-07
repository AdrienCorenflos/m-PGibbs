import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

kalman_rate = 2.601111111111111263e-01
Ns = [16, 32, 64, 128, 256]
pmmh_res = np.loadtxt("pmmh_results.txt")
current_gibbs = np.loadtxt("pgibbs_results_current.txt")
proposal_gibbs = np.loadtxt("pgibbs_results_proposal.txt")
mixture_gibbs = np.loadtxt("pgibbs_results_mixture.txt")

df = pd.DataFrame(columns=["PMMH", "Gibbs (v1)", "Gibbs (v2)", "Gibbs (v3)"],
                  index=Ns)

df["PMMH"] = pmmh_res
df["Gibbs (v1)"] = current_gibbs
df["Gibbs (v2)"] = proposal_gibbs
df["Gibbs (v3)"] = mixture_gibbs
df.index.name = "N"
df.columns.name = "Method"

fig, ax = plt.subplots(figsize=(8, 5))
df.plot(ax=ax, marker="o", markersize=8, fontsize=14)
ax.axhline(kalman_rate, color="k", linestyle="--", label="Ideal sampler")
ax.set_xlabel("N", fontsize=16)
ax.set_title("Acceptance Rate vs N", fontsize=16)
ax.legend(fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid()
plt.tight_layout()
plt.savefig("acceptance_rate.pdf")