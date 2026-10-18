"""CPLEX MIP baseline (docplex.mp).

Same 0-1 model as the Gurobi baseline. The pip CPLEX is the community edition,
capped at 1000 variables and 1000 constraints, so only 9x9 fits; larger grids
return LICENSE_LIMIT.
"""
import time

from docplex.mp.model import Model

from .common import (ERROR, INFEASIBLE, LICENSE_LIMIT, SOLVED, TIMEOUT, BaselineResult, Givens,
                     looks_like_license_limit)

NAME = "cplex_mip"


def solve(givens: Givens, box_size: int = 3, time_limit_s: float = 60.0) -> BaselineResult:
    n = box_size * box_size
    t0 = time.perf_counter()
    try:
        m = Model(name="sudoku")
        m.context.cplex_parameters.threads = 1
        m.set_time_limit(time_limit_s)
        x = m.binary_var_dict(((r, c, v) for r in range(n) for c in range(n) for v in range(n)), name="x")

        for r in range(n):
            for c in range(n):
                m.add_constraint(m.sum(x[r, c, v] for v in range(n)) == 1)
        for v in range(n):
            for r in range(n):
                m.add_constraint(m.sum(x[r, c, v] for c in range(n)) == 1)
            for c in range(n):
                m.add_constraint(m.sum(x[r, c, v] for r in range(n)) == 1)
            for br in range(0, n, box_size):
                for bc in range(0, n, box_size):
                    m.add_constraint(m.sum(
                        x[r, c, v]
                        for r in range(br, br + box_size) for c in range(bc, bc + box_size)) == 1)
        for r, c, v in givens:
            m.add_constraint(x[r, c, v] == 1)
        t_build = time.perf_counter() - t0

        t0 = time.perf_counter()
        sol = m.solve(log_output=False)
        t_solve = time.perf_counter() - t0
    except Exception as e:  # docplex wraps CPLEX errors in several exception types
        msg = str(e)
        status = LICENSE_LIMIT if looks_like_license_limit(msg) else ERROR
        return BaselineResult(status, None, time.perf_counter() - t0, 0.0, msg[:300])

    if sol is not None:
        grid = [[next(v + 1 for v in range(n) if sol.get_value(x[r, c, v]) > 0.5) for c in range(n)]
                for r in range(n)]
        return BaselineResult(SOLVED, grid, t_build, t_solve)
    details = m.get_solve_details()
    text = (details.status if details is not None else "").lower()
    if "infeasible" in text:
        return BaselineResult(INFEASIBLE, None, t_build, t_solve)
    if "limit" in text:
        return BaselineResult(TIMEOUT, None, t_build, t_solve)
    return BaselineResult(ERROR, None, t_build, t_solve, text)
