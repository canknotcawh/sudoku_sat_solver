"""Solve a Sudoku CNF with a PySAT-backed SAT solver."""
from dataclasses import dataclass
import time
from typing import List, Optional

from pysat.solvers import Solver

from .sudoku_encoder import SudokuEncoder, Clause


@dataclass
class StreamingSolveResult:
    grid: Optional[List[List[int]]]
    t_encode: float
    t_solver_load: float
    t_solve: float
    clause_count: int
    variable_count: int


def solve_streaming(encoder: SudokuEncoder, givens, solver_name: str = "cadical195") -> StreamingSolveResult:
    """Generate clauses incrementally and add each directly to a PySAT solver.

    ``t_encode`` measures Python clause generation; ``t_solver_load`` measures
    calls that transfer generated clauses into PySAT. The legacy materialized
    ``solve`` API is unchanged. Encode and load measurements are interleaved at
    clause granularity and their sum approximates pre-solve elapsed time.
    """
    solver = Solver(name=solver_name, use_timer=True)
    t_encode = 0.0
    t_load = 0.0
    count = 0
    clauses = iter(encoder.iter_encode(givens))
    try:
        while True:
            started = time.perf_counter()
            try:
                clause = next(clauses)
            except StopIteration:
                t_encode += time.perf_counter() - started
                break
            t_encode += time.perf_counter() - started
            started = time.perf_counter()
            solver.add_clause(clause)
            t_load += time.perf_counter() - started
            count += 1

        t0 = time.perf_counter()
        sat = solver.solve()
        t_solve = time.perf_counter() - t0
        grid = encoder.decode(solver.get_model()) if sat else None
        return StreamingSolveResult(grid, t_encode, t_load, t_solve,
                                    count, encoder.last_variable_count)
    finally:
        solver.delete()


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

    The blocking clause only needs to negate the n^3 cell variables p(r,c,v):
    any different grid must change at least one selected cell value. Auxiliary
    variables are intentionally omitted, even when an encoding admits multiple
    auxiliary assignments for the same grid.

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
