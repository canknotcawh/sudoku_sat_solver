"""CP-SAT Sudoku baseline with explicit status and timing information."""
from dataclasses import dataclass
import time
from typing import List, Optional, Tuple

Givens = List[Tuple[int, int, int]]


@dataclass
class CpSatResult:
    status: str
    grid: Optional[List[List[int]]]
    t_encode: float
    t_solve: float
    n_vars: int
    n_constraints: int


def solve_cp_sat_detailed(givens: Givens, box_size: int = 3, time_limit_s: float = 60.0) -> CpSatResult:
    """Build and solve one CP-SAT model; preserve timeout/infeasible status."""
    from ortools.sat.python import cp_model

    if isinstance(box_size, bool) or not isinstance(box_size, int) or box_size <= 0:
        raise ValueError("box_size must be a positive integer")
    if time_limit_s <= 0:
        raise ValueError("time_limit_s must be positive")
    n = box_size * box_size

    started = time.perf_counter()
    model = cp_model.CpModel()
    cells = [[model.NewIntVar(1, n, f"c_{r}_{c}") for c in range(n)] for r in range(n)]
    for row in cells:
        model.AddAllDifferent(row)
    for c in range(n):
        model.AddAllDifferent(cells[r][c] for r in range(n))
    for br in range(0, n, box_size):
        for bc in range(0, n, box_size):
            model.AddAllDifferent(
                cells[r][c]
                for r in range(br, br + box_size)
                for c in range(bc, bc + box_size)
            )
    for r, c, v in givens:
        model.Add(cells[r][c] == v + 1)
    t_encode = time.perf_counter() - started

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    started = time.perf_counter()
    status_code = solver.Solve(model)
    t_solve = time.perf_counter() - started
    status = solver.StatusName(status_code)
    if status_code in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        grid = [[solver.Value(cells[r][c]) for c in range(n)] for r in range(n)]
    else:
        grid = None
        if status_code == cp_model.UNKNOWN and t_solve >= time_limit_s:
            status = "TIMEOUT"
    n_constraints = 2 * n + box_size * box_size + len(givens)
    return CpSatResult(status, grid, t_encode, t_solve, n * n, n_constraints)


def solve_cp_sat(givens: Givens, box_size: int = 3, time_limit_s: float = 60.0):
    """Compatibility wrapper returning the grid or ``None``."""
    return solve_cp_sat_detailed(givens, box_size, time_limit_s).grid
