"""Encode a Sudoku puzzle (box_size x box_size boxes, n = box_size^2 grid) into CNF.

Variable convention: var(r, c, v) means "cell (r, c) holds value v" (0-indexed
r, c, v). Variables are numbered 1..n^3 in row-major (r, c, v) order; any
auxiliary variables needed by the chosen AMO encoding are numbered after that.
"""
from typing import Callable, List, Tuple

from .encodings.alo import alo
from .encodings.amo import AMO_ENCODERS

Clause = List[int]
Givens = List[Tuple[int, int, int]]


class VarPool:
    """Hands out fresh, increasing variable ids starting at `start`."""

    def __init__(self, start: int):
        self._next = start

    def new(self) -> int:
        v = self._next
        self._next += 1
        return v

    @property
    def count(self) -> int:
        return self._next - 1


class SudokuEncoder:
    def __init__(self, box_size: int = 3, amo_method: str = "pairwise"):
        if amo_method not in AMO_ENCODERS:
            raise ValueError(f"Unknown AMO encoding '{amo_method}', choose from {list(AMO_ENCODERS)}")
        self.box = box_size
        self.n = box_size * box_size
        self.amo_method = amo_method
        self._amo_fn = AMO_ENCODERS[amo_method]

    def var(self, r: int, c: int, v: int) -> int:
        n = self.n
        return r * n * n + c * n + v + 1

    def _amo(self, lits: List[int], pool: VarPool) -> List[Clause]:
        if self.amo_method == "pairwise":
            return self._amo_fn(lits)
        return self._amo_fn(lits, pool.new)

    def encode(self, givens: Givens) -> Tuple[List[Clause], int]:
        """Returns (clauses, total_variable_count)."""
        n, box = self.n, self.box
        clauses: List[Clause] = []
        pool = VarPool(start=n * n * n + 1)  # aux vars start right after the n^3 cell variables

        # 1. Each cell holds at least one value, and at most one value.
        for r in range(n):
            for c in range(n):
                lits = [self.var(r, c, v) for v in range(n)]
                clauses.extend(alo(lits))
                clauses.extend(self._amo(lits, pool))

        # 2. Each value appears at least once and at most once per row, column, box.
        for v in range(n):
            for r in range(n):
                lits = [self.var(r, c, v) for c in range(n)]
                clauses.extend(alo(lits))
                clauses.extend(self._amo(lits, pool))
            for c in range(n):
                lits = [self.var(r, c, v) for r in range(n)]
                clauses.extend(alo(lits))
                clauses.extend(self._amo(lits, pool))
            for br in range(0, n, box):
                for bc in range(0, n, box):
                    lits = [
                        self.var(r, c, v)
                        for r in range(br, br + box)
                        for c in range(bc, bc + box)
                    ]
                    clauses.extend(alo(lits))
                    clauses.extend(self._amo(lits, pool))

        # 3. Given clues become unit clauses.
        for (r, c, v) in givens:
            clauses.append([self.var(r, c, v)])

        return clauses, pool.count

    def decode(self, model: List[int]) -> List[List[int]]:
        """Turn a satisfying assignment (list of signed literals from the solver)
        back into an n x n grid of 1-indexed values."""
        true_lits = {lit for lit in model if lit > 0}
        n = self.n
        grid = [[0] * n for _ in range(n)]
        for r in range(n):
            for c in range(n):
                for v in range(n):
                    if self.var(r, c, v) in true_lits:
                        grid[r][c] = v + 1
                        break
        return grid

    def blocking_clause(self, grid: List[List[int]]) -> Clause:
        """A clause that is false exactly for this one grid: forbids the exact
        combination of (r, c, v) that produced it, so adding it and solving
        again finds a *different* solution if one exists. Used for the
        unique-solution check below."""
        n = self.n
        return [-self.var(r, c, grid[r][c] - 1) for r in range(n) for c in range(n)]
