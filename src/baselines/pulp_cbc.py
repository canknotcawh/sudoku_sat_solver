"""Open-source ILP baseline: PuLP with the bundled CBC solver.

Added because the free Gurobi and CPLEX editions cap model size. CBC has no
size limit, so this is the ILP reference for the large grids. Same 0-1 model
as the Gurobi and CPLEX MIP baselines.
"""
import time

import pulp

from .common import ERROR, INFEASIBLE, SOLVED, TIMEOUT, BaselineResult, Givens

NAME = "cbc_ilp"


def solve(givens: Givens, box_size: int = 3, time_limit_s: float = 60.0) -> BaselineResult:
    n = box_size * box_size
    t0 = time.perf_counter()
    prob = pulp.LpProblem("sudoku", pulp.LpMinimize)
    idx = range(n)
    x = [[[pulp.LpVariable(f"x_{r}_{c}_{v}", cat="Binary") for v in idx] for c in idx] for r in idx]
    prob += 0  # feasibility only

    for r in idx:
        for c in idx:
            prob += pulp.lpSum(x[r][c][v] for v in idx) == 1
    for v in idx:
        for r in idx:
            prob += pulp.lpSum(x[r][c][v] for c in idx) == 1
        for c in idx:
            prob += pulp.lpSum(x[r][c][v] for r in idx) == 1
        for br in range(0, n, box_size):
            for bc in range(0, n, box_size):
                prob += pulp.lpSum(
                    x[r][c][v]
                    for r in range(br, br + box_size) for c in range(bc, bc + box_size)) == 1
    for r, c, v in givens:
        prob += x[r][c][v] == 1
    t_build = time.perf_counter() - t0

    t0 = time.perf_counter()
    try:
        prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit_s, threads=1))
    except Exception as e:
        return BaselineResult(ERROR, None, t_build, time.perf_counter() - t0, str(e)[:300])
    t_solve = time.perf_counter() - t0

    status = pulp.LpStatus[prob.status]
    if status == "Optimal":
        grid = [[next(v + 1 for v in idx if (x[r][c][v].value() or 0) > 0.5) for c in idx] for r in idx]
        return BaselineResult(SOLVED, grid, t_build, t_solve)
    if status == "Infeasible":
        return BaselineResult(INFEASIBLE, None, t_build, t_solve)
    return BaselineResult(TIMEOUT, None, t_build, t_solve, status)
