"""
Tests for the Schelling model. Run with:  python -m pytest tests   (or  python tests/test_model.py)
"""
import os
import random
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "legacy"))

import segregationModel as original  # noqa: E402
from schelling import EXTENDED_12, MOORE_8, Model, constant_tolerance  # noqa: E402

CASES = [(0.1, 0.1), (0.3, 0.3), (0.45, 0.45), (0.6, 0.6), (0.2, 0.6), (0.5, 0.7), (0.9, 0.9)]


def model(tx, ty, nb=EXTENDED_12, seed=0):
    return Model(50, 10, ["#", "O"], [0.45, 0.45],
                 [constant_tolerance(tx), constant_tolerance(ty)], nb, seed=seed)


def count_unsatisfied(m):
    return sum(not c.is_satisfied(m.neighborhood) for c in m.grid.agents())


def test_legacy_mode_reproduces_original_code():
    """legacy_cache=True must give the original code's exact output and final grid."""
    for nb in (EXTENDED_12, MOORE_8):
        for tx, ty in CASES:
            for seed in range(3):
                random.seed(seed)
                old = original.Model(50, 10, ["#", "O"], [0.45, 0.45],
                                     [lambda q, t=tx: (t, 1.0), lambda q, t=ty: (t, 1.0)], nb)
                expected = old.run(3000)
                new = model(tx, ty, nb, seed)
                assert new.run(3000, legacy_cache=True) == expected
                assert new.grid.visualize().strip() == old.grid.visualize().strip()


def test_converged_runs_leave_no_unsatisfied_agent():
    for tx, ty in CASES[:4]:
        for seed in range(3):
            m = model(tx, ty, seed=seed)
            _, complete, _, _ = m.run(20_000)
            assert complete
            assert count_unsatisfied(m) == 0


def test_cache_matches_brute_force_at_any_stopping_point():
    """Stop runs early at random points: the cached count must equal a full recount."""
    rng = random.Random(42)
    for _ in range(40):
        tx, ty = rng.random(), rng.random()
        m = model(tx, ty, seed=rng.randrange(10_000))
        m.run(rng.randint(1, 1500))
        assert m.unsatisfied_at_end == count_unsatisfied(m)


def test_seed_makes_runs_reproducible():
    a, b = model(0.5, 0.5, seed=3), model(0.5, 0.5, seed=3)
    assert a.run(5000) == b.run(5000)


def test_input_validation():
    for bad in ([0.6, 0.6], [-0.1, 0.5]):
        try:
            Model(10, 10, ["#", "O"], bad, [constant_tolerance(0.3)] * 2, MOORE_8)
        except ValueError:
            continue
        raise AssertionError(f"densities {bad} should be rejected")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("passed:", name)
