"""CP-SAT (OR-Tools) baseline — for comparing against the SAT-encoding approach.

Uses AllDifferent constraints directly instead of Boolean SAT variables, which
is the natural CP formulation and a fair point of comparison for the report.
"""
from typing import List, Optional, Tuple

from ortools.sat.python import cp_model

Givens = List[Tuple[int, int, int]]


def solve_cp_sat(givens: Givens, box_size: int = 3, time_limit_s: float = 60.0) -> Optional[List[List[int]]]:
    n = box_size * box_size
    model = cp_model.CpModel()
    cells = [[model.NewIntVar(1, n, f"c_{r}_{c}") for c in range(n)] for r in range(n)]

    for row in cells:
        model.AddAllDifferent(row)
    for c in range(n):
        model.AddAllDifferent(cells[r][c] for r in range(n))
    for br in range(0, n, box_size):
        for bc in range(0, n, box_size):
            model.AddAllDifferent(
                cells[r][c] for r in range(br, br + box_size) for c in range(bc, bc + box_size)
            )
    for (r, c, v) in givens:
        model.Add(cells[r][c] == v + 1)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None
    return [[solver.Value(cells[r][c]) for c in range(n)] for r in range(n)]


# TODO: ILP baseline (Gurobi or CPLEX MIP) belongs alongside this file, e.g.
# src/baselines/ilp_solver.py, using binary vars x[r][c][v] with sum-to-one
# constraints per cell/row/col/box — structurally close to the SAT encoding,
# which makes for a clean side-by-side comparison in the report.
