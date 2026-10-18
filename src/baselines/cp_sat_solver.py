"""CP-SAT (OR-Tools) baseline.

Constraint-programming formulation: one integer variable per cell and an
AllDifferent over every row, column and box. This is the natural CP model
and differs from the Boolean one-hot model used for SAT/ILP.
"""
import time

from ortools.sat.python import cp_model

from .common import ERROR, INFEASIBLE, SOLVED, TIMEOUT, BaselineResult, Givens

NAME = "cpsat"


def solve(givens: Givens, box_size: int = 3, time_limit_s: float = 60.0) -> BaselineResult:
    n = box_size * box_size
    t0 = time.perf_counter()
    model = cp_model.CpModel()
    cells = [[model.NewIntVar(1, n, f"c_{r}_{c}") for c in range(n)] for r in range(n)]
    for r in range(n):
        model.AddAllDifferent(cells[r])
    for c in range(n):
        model.AddAllDifferent([cells[r][c] for r in range(n)])
    for br in range(0, n, box_size):
        for bc in range(0, n, box_size):
            model.AddAllDifferent(
                [cells[r][c] for r in range(br, br + box_size) for c in range(bc, bc + box_size)]
            )
    for r, c, v in givens:
        model.Add(cells[r][c] == v + 1)
    t_build = time.perf_counter() - t0

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_workers = 1
    t0 = time.perf_counter()
    status = solver.Solve(model)
    t_solve = time.perf_counter() - t0

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        grid = [[solver.Value(cells[r][c]) for c in range(n)] for r in range(n)]
        return BaselineResult(SOLVED, grid, t_build, t_solve)
    if status == cp_model.INFEASIBLE:
        return BaselineResult(INFEASIBLE, None, t_build, t_solve)
    if status == cp_model.UNKNOWN:
        return BaselineResult(TIMEOUT, None, t_build, t_solve)
    return BaselineResult(ERROR, None, t_build, t_solve, f"cp_model status {status}")