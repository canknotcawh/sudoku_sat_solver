"""Solve a Sudoku CNF with a PySAT-backed SAT solver."""
from dataclasses import dataclass
from typing import List, Optional

from pysat.solvers import Solver

from .sudoku_encoder import SudokuEncoder, Clause


def solve(encoder: SudokuEncoder, clauses: List[Clause], solver_name: str = "cadical195") -> Optional[List[List[int]]]:
    """Returns the solved n x n grid, or None if the CNF is UNSAT.

    `solver_name` must be one of the names PySAT accepts — see
    `pysat.solvers.SolverNames` for the exact strings bundled in your
    installed python-sat version (commonly cadical195, glucose42, minisat22,
    cryptominisat, lingeling, maplechrono, ...).
    """
    with Solver(name=solver_name, bootstrap_with=clauses, use_timer=True) as s:
        if not s.solve():
            return None
        model = s.get_model()
        return encoder.decode(model)


@dataclass
class UniquenessResult:
    grid: Optional[List[List[int]]]       # None if UNSAT
    is_unique: Optional[bool]             # None if UNSAT (uniqueness is moot)
    second_grid: Optional[List[List[int]]]  # a second, different solution, if one exists


def solve_check_unique(
    encoder: SudokuEncoder,
    clauses: List[Clause],
    solver_name: str = "cadical195",
) -> UniquenessResult:
    """Solve, then check whether the solution is the only one.

    Standard trick for "is this SAT model unique": after finding a model,
    add a *blocking clause* that forbids exactly that assignment, and ask the
    solver to solve again on the same instance. If it comes back UNSAT, no
    other model exists, so the first one was unique.

    The blocking clause only needs to negate the n^3 cell variables p(r,c,v)
    (one of the var(r,c,v) literals must flip for the grid to actually
    differ) — auxiliary variables introduced by sequential/commander/binary/
    product AMO encodings are functions of those, not independent choices,
    so leaving them out of the blocking clause still blocks exactly this one
    grid, not fewer. This also means the check works the same way regardless
    of which AMO encoding produced `clauses`.

    Uses one incremental solver instance (`add_clause` + solve again) so the
    second call reuses learned clauses instead of starting from scratch.
    """
    n = encoder.n
    cell_vars = {encoder.var(r, c, v) for r in range(n) for c in range(n) for v in range(n)}

    with Solver(name=solver_name, bootstrap_with=clauses, use_timer=True) as s:
        if not s.solve():
            return UniquenessResult(grid=None, is_unique=None, second_grid=None)

        model = s.get_model()
        grid = encoder.decode(model)

        blocking_clause = [-lit for lit in model if lit > 0 and lit in cell_vars]
        s.add_clause(blocking_clause)

        if not s.solve():
            return UniquenessResult(grid=grid, is_unique=True, second_grid=None)

        second_model = s.get_model()
        second_grid = encoder.decode(second_model)
        return UniquenessResult(grid=grid, is_unique=False, second_grid=second_grid)
