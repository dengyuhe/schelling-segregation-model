"""
Agent-based Schelling segregation model on a finite 2-D grid.

Each agent has a symbol (its group) and a tolerance band [min_same, max_same]:
it is *satisfied* when the share of same-group agents among its occupied
neighbors lies inside that band. At every step, one unsatisfied agent is
picked at random and moved to a random empty cell. The run ends when every
agent is satisfied (convergence) or the move budget is exhausted.

In the write-up, ``min_same`` is called the "minimum tolerance": the minimum
share of same-type neighbors an agent requires. A higher value therefore means
a *less* tolerant agent.
"""

import random

import numpy as np

# --- Neighborhoods ------------------------------------------------------------
# Offsets (dx, dy) that define which cells an agent looks at.
VON_NEUMANN_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MOORE_8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
EXTENDED_12 = MOORE_8 + [(2, 0), (0, 2), (-2, 0), (0, -2)]   # used in all experiments

NEIGHBORHOODS = {4: VON_NEUMANN_4, 8: MOORE_8, 12: EXTENDED_12}


def constant_tolerance(min_same, max_same=1.0):
    """Tolerance function giving every agent of a group the same band."""
    return lambda quantile: (min_same, max_same)


# --- Agents and grid -------------------------------------------------------------
class Cell:
    """One agent: a group symbol, a tolerance band, and its position on a grid."""

    def __init__(self, symbol, min_same, max_same=1.0):
        self.symbol = symbol
        self.min = min_same
        self.max = max_same
        self.grid = None
        self.x = None
        self.y = None
        self.moves = 0          # number of times this agent has relocated

    def get_relative(self, dx, dy):
        return self.grid.get_cell(self.x + dx, self.y + dy)

    def neighbor_counts(self, neighborhood):
        """Return (same-group neighbors, occupied neighbors)."""
        g = self.grid
        cells, dim_x, dim_y, x0, y0 = g.cells, g.dim_x, g.dim_y, self.x, self.y
        same = total = 0
        for dx, dy in neighborhood:            # inlined get_relative() for speed
            x, y = x0 + dx, y0 + dy
            if 0 <= x < dim_x and 0 <= y < dim_y:
                n = cells[x][y]
                if n is not None:
                    total += 1
                    same += n.symbol == self.symbol
        return same, total

    def neighborhood_ratio(self, neighborhood):
        """Share of occupied neighbors in the same group (1.0 if no neighbors)."""
        same, total = self.neighbor_counts(neighborhood)
        return 1.0 if total == 0 else same / total

    def is_satisfied(self, neighborhood):
        return self.min <= self.neighborhood_ratio(neighborhood) <= self.max


class Grid:
    """A dim_x by dim_y lattice (no wrap-around) holding Cells or empty slots."""

    def __init__(self, dim_x, dim_y, rng=None):
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.rng = rng or random.Random()
        self.cells = [[None for _ in range(dim_y)] for _ in range(dim_x)]
        self.empties = [(x, y) for x in range(dim_x) for y in range(dim_y)]   # cache of empty slots

    def add_cell(self, x, y, cell):
        self.cells[x][y] = cell
        self.empties.remove((x, y))
        if cell is None:
            self.empties.append((x, y))
        else:
            cell.grid, cell.x, cell.y = self, x, y

    def get_cell(self, x, y):
        if 0 <= x < self.dim_x and 0 <= y < self.dim_y:
            return self.cells[x][y]
        return None

    def move_to(self, xi, yi, xf, yf):
        if (xf, yf) not in self.empties:
            raise ValueError(f"Cannot move a cell to non-empty coordinates {xf}, {yf}")
        c = self.cells[xi][yi]
        self.cells[xf][yf], self.cells[xi][yi] = c, None
        self.empties.remove((xf, yf))
        self.empties.append((xi, yi))
        c.x, c.y = xf, yf
        c.moves += 1

    def move_to_random_empty(self, xi, yi):
        xf, yf = self.empties[self.rng.randint(0, len(self.empties) - 1)]
        self.move_to(xi, yi, xf, yf)
        return xf, yf

    def agents(self):
        for x in range(self.dim_x):
            for y in range(self.dim_y):
                if self.cells[x][y] is not None:
                    yield self.cells[x][y]

    def sameness_index(self, neighborhood):
        """Average same-group share across all agents (the model's segregation measure)."""
        total, n = 0.0, 0
        for y in range(self.dim_y):          # row-major order, matching the original code
            for x in range(self.dim_x):
                c = self.cells[x][y]
                if c is not None:
                    total += c.neighborhood_ratio(neighborhood)
                    n += 1
        return total / n if n else None

    def to_array(self, symbols):
        """Grid as an int array: 0 = empty, k = k-th symbol (1-based). Shape (dim_y, dim_x)."""
        code = {s: i + 1 for i, s in enumerate(symbols)}
        a = np.zeros((self.dim_y, self.dim_x), dtype=np.int8)
        for c in self.agents():
            a[c.y, c.x] = code[c.symbol]
        return a

    def visualize(self):
        return "\n".join(
            "".join(self.cells[x][y].symbol if self.cells[x][y] else "-" for x in range(self.dim_x))
            for y in range(self.dim_y))


# --- Model -------------------------------------------------------------------------
class Model:
    """
    Parameters
    ----------
    dim_x, dim_y     : grid dimensions
    symbols          : group labels, e.g. ["#", "O"]
    densities        : share of the grid occupied by each group (each in [0, 1], sum <= 1);
                       the remainder is left empty
    tolerance_funcs  : one function per group mapping an agent's quantile within its group
                       (0 to 1) to its (min_same, max_same) band; allows heterogeneous agents
    neighborhood     : list of (dx, dy) offsets, e.g. EXTENDED_12
    seed             : optional random seed for reproducibility
    """

    def __init__(self, dim_x, dim_y, symbols, densities, tolerance_funcs, neighborhood, seed=None):
        if not len(symbols) == len(densities) == len(tolerance_funcs):
            raise ValueError(f"Got {len(symbols)} symbols, {len(densities)} densities and "
                             f"{len(tolerance_funcs)} tolerance functions; they must match.")
        if any(not 0 <= d <= 1 for d in densities):
            raise ValueError(f"Densities {densities} must each lie in [0, 1].")
        if sum(densities) > 1:
            raise ValueError(f"Densities {densities} sum to more than 1.")

        self.rng = random.Random(seed)
        self.symbols = list(symbols)
        self.neighborhood = list(neighborhood)
        self.is_run = False

        # Create agents; each group's agents get tolerances by their quantile in the group
        n = dim_x * dim_y
        counts = [int(c) for c in np.floor(np.multiply(densities, n))]
        pool = []
        for symbol, count, func in zip(symbols, counts, tolerance_funcs):
            for j in range(count):
                q = j / (count - 1) if count > 1 else 0.0
                lo, hi = func(q)
                pool.append(Cell(symbol, lo, hi))
        pool += [None] * (n - sum(counts))

        # Place agents and empty slots uniformly at random
        self.grid = Grid(dim_x, dim_y, self.rng)
        for x in range(dim_x):
            for y in range(dim_y):
                self.grid.add_cell(x, y, pool.pop(self.rng.randint(0, len(pool) - 1)))

    # ---------------------------------------------------------------------------
    def run(self, max_iterations, record=False, trace_every=100, legacy_cache=False, visualize=False):
        """
        Run until every agent is satisfied or ``max_iterations`` loop iterations have passed.

        Returns (iterations, complete, initial_sameness, final_sameness), the same tuple as
        the original code. ``complete`` is True when no unsatisfied agent remains.

        record=True stores diagnostics in ``self.history``:
            sameness       : list of (move number, sameness index), every ``trace_every`` moves
            mover_symbols  : group of the agent moved at each step
            origin_counts  : array (dim_y, dim_x), how many moves started at each cell
            frames         : grid snapshots (as int arrays) every ``trace_every`` moves

        legacy_cache=True reproduces the original implementation exactly (same random draws,
        same output). After a move it re-checks only the 3x3 box around the destination,
        missing the neighbors of the vacated cell and cells two steps away in a 12-cell
        neighborhood, so a run can report convergence while some agents are still
        unsatisfied. The default (False) re-checks every affected agent.
        """
        if self.is_run:
            raise ValueError("This model has already been run; create a new Model.")
        self.is_run = True
        g, nb = self.grid, self.neighborhood
        initial_sameness = g.sameness_index(nb)

        if record:
            self.history = dict(sameness=[(0, initial_sameness)], mover_symbols=[],
                                origin_counts=np.zeros((g.dim_y, g.dim_x), dtype=int),
                                frames=[g.to_array(self.symbols)])

        # Cache of unsatisfied agents' positions: a list for O(1) random choice
        # plus a dict (position -> index in list) for O(1) removal.
        unhappy = [(x, y) for x in range(g.dim_x) for y in range(g.dim_y)
                   if g.cells[x][y] is not None and not g.cells[x][y].is_satisfied(nb)]
        index = {p: i for i, p in enumerate(unhappy)}

        def discard(p):
            i = index.pop(p, None)
            if i is None:
                return
            last = unhappy.pop()
            if i < len(unhappy):
                unhappy[i] = last
                index[last] = i

        def refresh(p):
            if not (0 <= p[0] < g.dim_x and 0 <= p[1] < g.dim_y):
                return
            discard(p)
            c = g.cells[p[0]][p[1]]
            if c is not None and not c.is_satisfied(nb):
                index[p] = len(unhappy)
                unhappy.append(p)

        # Cells whose satisfaction can change when cell (x, y) changes occupancy
        affected_offsets = sorted({(-dx, -dy) for dx, dy in nb})

        iterations, complete, moves = 0, False, 0
        for _ in range(max_iterations):
            iterations += 1
            if not unhappy:
                complete = True
                break

            i = self.rng.randint(0, len(unhappy) - 1)
            xi, yi = unhappy[i]
            mover = g.get_cell(xi, yi)
            xf, yf = g.move_to_random_empty(xi, yi)
            moves += 1

            if legacy_cache:
                # Original update rule, kept verbatim for reproducibility
                unhappy.pop(i)
                for x in range(xf - 1, xf + 2):
                    for y in range(yf - 1, yf + 2):
                        if (x, y) in unhappy:
                            unhappy.remove((x, y))
                        c = g.get_cell(x, y)
                        if c is not None and not c.is_satisfied(nb):
                            unhappy.append((x, y))
            else:
                discard((xi, yi))
                for dx, dy in affected_offsets:
                    refresh((xi + dx, yi + dy))
                    refresh((xf + dx, yf + dy))
                refresh((xf, yf))

            if record:
                self.history["mover_symbols"].append(mover.symbol)
                self.history["origin_counts"][yi, xi] += 1
                if moves % trace_every == 0:
                    self.history["sameness"].append((moves, g.sameness_index(nb)))
                    self.history["frames"].append(g.to_array(self.symbols))

        if legacy_cache:
            # In legacy mode the cache can be empty while agents are still unsatisfied
            self.unsatisfied_at_end = sum(not c.is_satisfied(nb) for c in g.agents())
        else:
            self.unsatisfied_at_end = len(unhappy)
        self.moves = moves
        final_sameness = g.sameness_index(nb)
        if record:
            self.history["sameness"].append((moves, final_sameness))
            self.history["frames"].append(g.to_array(self.symbols))
        if visualize:
            print(g.visualize())
        return iterations, complete, initial_sameness, final_sameness
