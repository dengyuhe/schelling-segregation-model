"""
Animated GIFs on a 50 x 50 grid: a mild (30%) and a strong (70%) same-group
preference. Each group fills 42.5% of cells, 15% empty, 8-cell neighborhood.

    python experiments/animate.py
"""
import os

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap

from common import FIGURES, MOORE_8, Model, constant_tolerance

cmap = ListedColormap(["white", "#d62728", "#1f4fd8"])
RECORD_EVERY = 5      # snapshot frequency (moves)
N_FRAMES = 90         # frames shown, evenly spaced over the run

for pct in (30, 70):
    t = pct / 100
    m = Model(50, 50, ["#", "O"], [0.425, 0.425], [constant_tolerance(t)] * 2, MOORE_8, seed=7)
    m.run(200_000, record=True, trace_every=RECORD_EVERY)
    snaps = m.history["frames"]
    picks = sorted({round(i * (len(snaps) - 1) / (N_FRAMES - 1)) for i in range(N_FRAMES)})
    frames = [snaps[i] for i in picks]
    moves_at = [min(i * RECORD_EVERY, m.moves) for i in picks]
    n_real = len(frames)
    final_same = m.history["sameness"][-1][1]
    frames = frames + [frames[-1]] * 15             # hold the final state

    fig, ax = plt.subplots(figsize=(4, 4.6))
    im = ax.imshow(frames[0], cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    title = ax.set_title("", fontsize=10)

    def update(k):
        im.set_data(frames[k])
        done = k >= n_real - 1
        mv = m.moves if done else moves_at[k]
        title.set_text(f"Each agent wants at least {pct}% similar neighbors\n"
                       f"moves: {mv:,}" + (f"   final sameness: {final_same:.2f}" if done else ""))
        return im, title

    anim = FuncAnimation(fig, update, frames=len(frames), interval=80)
    fig.tight_layout()
    out = os.path.join(FIGURES, f"animation_{pct}pct.gif")
    anim.save(out, writer=PillowWriter(fps=12))
    plt.close(fig)
    print("saved", out, f"({m.moves} moves, frames {n_real}, all satisfied={m.unsatisfied_at_end == 0})")
