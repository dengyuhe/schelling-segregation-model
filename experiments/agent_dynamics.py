"""
Why the regions behave differently: per-agent diagnostics for single runs.

  - moves per agent, by group: converging runs show a heavy-tailed histogram,
    non-converging groups a bell-shaped one
  - sameness index over time, with the share of recent moves made by O agents
  - movement heatmap: where moves start on the grid

    python experiments/agent_dynamics.py [--legacy]
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from common import FIGURES, MAX_ITERATIONS, make_model

p = argparse.ArgumentParser()
p.add_argument("--legacy", action="store_true")
a = p.parse_args()
tag = "legacy" if a.legacy else "fixed"

POINTS = {"Region II (0.6, 0.6)": (0.6, 0.6),
          "Region III (0.9, 0.9)": (0.9, 0.9),
          "Region IV (0.2, 0.6)": (0.2, 0.6)}

runs = {}
for name, (x, y) in POINTS.items():
    m = make_model(x, y, seed=1)
    m.run(MAX_ITERATIONS, record=True, trace_every=50, legacy_cache=a.legacy)
    m.grid_tolerance = {"#": x, "O": y}
    runs[name] = m

fig, axes = plt.subplots(3, 3, figsize=(14, 10))
for col, (name, m) in enumerate(runs.items()):
    h = m.history
    ax = axes[0, col]
    all_moves = [c.moves for c in m.grid.agents()]
    bins = np.linspace(0, max(all_moves) + 1, 41)
    for sym, color in [("#", "#8e44ad"), ("O", "#16a085")]:
        mv = [c.moves for c in m.grid.agents() if c.symbol == sym]
        tol = m.grid_tolerance[sym]
        ax.hist(mv, bins=bins, histtype="step", lw=1.6, color=color,
                label=f"{sym} agents (min tolerance {tol})")
    ax.set_yscale("log")
    ax.set(title=name, xlabel="Moves made per agent", ylabel="Agents")
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1, col]
    t, s = zip(*h["sameness"])
    ax.plot(t, s, ".", color="#1f4fd8", label="Sameness index")
    movers = np.array([c == "O" for c in h["mover_symbols"]], float)
    if len(movers) >= 100:
        share = np.convolve(movers, np.ones(100) / 100, mode="valid")
        ax.plot(np.arange(100, len(movers) + 1), share, color="#c71585", lw=0.8,
                label="Share of last 100 moves by O")
    ax.set(xlabel="Move number", ylim=(0, 1.05))
    ax.legend(frameon=False, fontsize=8)
    ax = axes[2, col]
    ax.imshow(h["origin_counts"], cmap="magma", origin="lower", aspect="auto")
    ax.set(title="Where moves start", xlabel="x", ylabel="y")
for ax in axes.flat:
    ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(FIGURES, f"agent_dynamics_{tag}.png"), dpi=140)
print("saved", f"figures/agent_dynamics_{tag}.png")
