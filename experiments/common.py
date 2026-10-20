"""Shared settings for the experiments: the cross-section studied in the write-up."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from schelling import EXTENDED_12, MOORE_8, Model, constant_tolerance  # noqa: E402,F401

ROOT = os.path.join(os.path.dirname(__file__), "..")
FIGURES = os.path.join(ROOT, "figures")
RESULTS = os.path.join(ROOT, "results")
os.makedirs(FIGURES, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)

# Fixed settings ("invariants") of the cross-section
DIM_X, DIM_Y = 50, 10            # 500 cells
DENSITIES = [0.45, 0.45]         # equal group sizes, 10% vacancies
SYMBOLS = ["#", "O"]
NEIGHBORHOOD = EXTENDED_12       # 12-cell neighborhood
MAX_ITERATIONS = 20_000          # soft cap: no convergent run in the write-up needed more


def make_model(min_hash, min_o, seed=None):
    """Model in the cross-section: only the two minimum tolerances vary."""
    return Model(DIM_X, DIM_Y, SYMBOLS, DENSITIES,
                 [constant_tolerance(min_hash), constant_tolerance(min_o)],
                 NEIGHBORHOOD, seed=seed)
