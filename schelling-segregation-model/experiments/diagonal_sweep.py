"""
Equal minimum tolerances for both groups (the diagonal X = Y of the cross-section).

Plots final sameness and the number of moves to convergence against the shared
tolerance. Moves grow roughly exponentially until the tipping point near 0.75,
beyond which runs stop converging.

    python experiments/diagonal_sweep.py [--seeds 3] [--step 0.01] [--legacy]
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from common import FIGURES, MAX_ITERATIONS, RESULTS, make_model

p = argparse.ArgumentParser()
p.add_argument("--seeds", type=int, default=3)
p.add_argument("--step", type=float, default=0.01)
p.add_argument("--legacy", action="store_true", help="use the original cache update rule")
a = p.parse_args()
tag = "legacy" if a.legacy else "fixed"

ts = np.round(np.arange(0, 1, a.step), 4)
rows = []
for t in ts:
    for seed in range(a.seeds):
        m = make_model(t, t, seed=seed)
        it, complete, s0, s1 = m.run(MAX_ITERATIONS, legacy_cache=a.legacy)
        rows.append((t, seed, m.moves, complete, s1))
r = np.array(rows, dtype=float)
np.save(os.path.join(RESULTS, f"diagonal_sweep_{tag}.npy"), r)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
ax1.plot(r[:, 0], r[:, 4], ".", ms=3, color="#1f4fd8")
ax1.set(xlabel="Minimum tolerance (both groups)", ylabel="Final sameness index",
        title="Segregation rises with tolerance, then collapses")
conv = r[:, 3] == 1
ax2.semilogy(r[conv, 0], np.maximum(r[conv, 2], 1), ".", ms=3, color="#c0392b", label="converged")
ax2.semilogy(r[~conv, 0], r[~conv, 2], ".", ms=3, color="#e6a700", label="hit the cap (did not converge)")
ax2.set(xlabel="Minimum tolerance (both groups)", ylabel="Moves (log scale)", title="Moves to equilibrium")
ax2.legend(frameon=False, fontsize=8)
for ax in (ax1, ax2):
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(FIGURES, f"diagonal_sweep_{tag}.png"), dpi=160)
print("saved", f"figures/diagonal_sweep_{tag}.png")
