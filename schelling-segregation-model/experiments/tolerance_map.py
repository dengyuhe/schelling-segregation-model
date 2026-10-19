"""
Two-dimensional cross-section: vary the minimum tolerance of each group independently.

For every (X, Y) pair, runs seeded trials and records
  - mean final sameness index        (blue map)
  - mean log10 number of moves       (red map)
  - share of trials that hit the cap (non-convergence; yellow contour)

    python experiments/tolerance_map.py [--step 0.04] [--seeds 1] [--legacy] [--workers 4]
    python experiments/tolerance_map.py --plot-only [--legacy]      # re-plot saved results

A fine map (step 0.01, several seeds) takes hours on one core; use --workers to parallelize.
"""
import argparse
import os
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from common import FIGURES, MAX_ITERATIONS, RESULTS, make_model


def run_point(x, y, seed, legacy=False):
    """One trial: (final sameness, log10 moves, 1 if it hit the cap else 0)."""
    m = make_model(x, y, seed=seed)
    _, complete, _, s1 = m.run(MAX_ITERATIONS, legacy_cache=legacy)
    return s1, np.log10(max(m.moves, 1)), 0.0 if complete else 1.0


def _job(args):
    i, j, x, y, seed, legacy = args
    return (i, j) + run_point(x, y, seed, legacy)


def compute(step, seeds, legacy=False, workers=1, cols=None):
    """Arrays indexed [row = O tolerance, col = # tolerance]. ``cols`` limits the # values computed."""
    grid = np.round(np.arange(0, 1, step), 4)
    n = len(grid)
    cols = range(n) if cols is None else cols
    same, logm, nonconv = (np.zeros((n, n)) for _ in range(3))
    jobs = [(i, j, grid[i], grid[j], s, legacy) for i in cols for j in range(n) for s in range(seeds)]
    with Pool(workers) as pool:
        for i, j, s1, lm, nc in pool.imap_unordered(_job, jobs, chunksize=2):
            same[j, i] += s1 / seeds
            logm[j, i] += lm / seeds
            nonconv[j, i] += nc / seeds
    return grid, same, logm, nonconv


def plot(path_npz, path_png):
    d = np.load(path_npz)
    grid, step = d["grid"], float(d["grid"][1] - d["grid"][0])
    blue = LinearSegmentedColormap.from_list("b", ["black", "#1f3fff"])
    red = LinearSegmentedColormap.from_list("r", ["black", "#ff2a2a"])
    ext = [0, grid[-1] + step, 0, grid[-1] + step]
    centers = grid + step / 2
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, key, cmap, title, label in [
            (axes[0], "log_moves", red, "Moves to equilibrium", "log10 moves (cap = 20,000)"),
            (axes[1], "sameness", blue, "Final sameness index", "sameness index")]:
        im = ax.imshow(d[key], origin="lower", extent=ext, cmap=cmap, interpolation="nearest")
        ax.contour(centers, centers, d["nonconverged"], levels=[0.5], colors="yellow", linewidths=1.2)
        ax.plot([0.5, 1.0], [1.0, 0.5], "w--", lw=0.8)          # X + Y = 1.5
        ax.set(xlabel="# minimum tolerance (X)", ylabel="O minimum tolerance (Y)", title=title,
               xlim=(0, 1), ylim=(0, 1))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=label)
    fig.text(0.5, 0.01, "Yellow: boundary between runs that converge and runs that hit the cap.  "
             "Dashed white: X + Y = 1.5", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path_png, dpi=160)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--step", type=float, default=0.04)
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--legacy", action="store_true", help="use the original cache update rule")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--plot-only", action="store_true")
    a = p.parse_args()
    tag = "legacy" if a.legacy else "fixed"
    npz = os.path.join(RESULTS, f"tolerance_map_{tag}.npz")
    if not a.plot_only:
        grid, same, logm, nonconv = compute(a.step, a.seeds, a.legacy, a.workers)
        np.savez(npz, grid=grid, sameness=same, log_moves=logm, nonconverged=nonconv)
    plot(npz, os.path.join(FIGURES, f"tolerance_map_{tag}.png"))
    print("saved", f"figures/tolerance_map_{tag}.png")
