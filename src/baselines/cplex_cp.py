"""CPLEX CP Optimizer baseline (docplex.cp), AllDifferent formulation.

The community edition of CP Optimizer rejects models whose search space is
above 2^1000, which in practice allows 9x9 and 16x16 but not 25x25; larger
grids return LICENSE_LIMIT.
"""
import time

from docplex.cp.model import CpoModel

from .common import (ERROR, INFEASIBLE, LICENSE_LIMIT, SOLVED, TIMEOUT, BaselineResult, Givens,
                     looks_like_license_limit)

NAME = "cplex_cp"


def solve(givens: Givens, box_size: int = 3, time_limit_s: float = 60.0) -> BaselineResult:
    n = box_size * box_size
    t0 = time.perf_counter()
    try:
        m = CpoModel()
        cells = [[m.integer_var(1, n, f"c_{r}_{c}") for c in range(n)] for r in range(n)]
        for r in range(n):
            m.add(m.all_diff(cells[r]))
        for c in range(n):
            m.add(m.all_diff([cells[r][c] for r in range(n)]))
        for br in range(0, n, box_size):
            for bc in range(0, n, box_size):
                m.add(m.all_diff([cells[r][c] for r in range(br, br + box_size)
                                  for c in range(bc, bc + box_size)]))
        for r, c, v in givens:
            m.add(cells[r][c] == v + 1)
        t_build = time.perf_counter() - t0

        t0 = time.perf_counter()
        res = m.solve(TimeLimit=time_limit_s, Workers=1, log_output=None)
        t_solve = time.perf_counter() - t0
    except Exception as e:
        msg = str(e)
        status = LICENSE_LIMIT if looks_like_license_limit(msg) else ERROR
        return BaselineResult(status, None, time.perf_counter() - t0, 0.0, msg[:300])

    if res.is_solution():
        grid = [[res.get_value(cells[r][c]) for c in range(n)] for r in range(n)]
        return BaselineResult(SOLVED, grid, t_build, t_solve)
    status_text = res.get_solve_status()
    if status_text == "Infeasible":
        return BaselineResult(INFEASIBLE, None, t_build, t_solve)
    return BaselineResult(TIMEOUT, None, t_build, t_solve, str(status_text))
