"""Gurobi ILP baseline.

Same one-hot model as the SAT encoding, written as 0-1 linear constraints:
x[r,c,v] = 1 iff cell (r,c) holds v, and every cell / row-value / column-value /
box-value sums to exactly 1. Pure feasibility problem (objective 0).

The pip `gurobipy` ships with a size-limited licence (about 2000 variables or
constraints), which fits 9x9 only. Larger grids return LICENSE_LIMIT; use the
free academic licence to lift it.
"""
import time

import gurobipy as gp
from gurobipy import GRB

from .common import (ERROR, INFEASIBLE, LICENSE_LIMIT, SOLVED, TIMEOUT, BaselineResult, Givens,
                     diagnostic, package_version)

NAME = "gurobi"
VERSION = ".".join(map(str, gp.gurobi.version()))


def classify_gurobi_error_code(error_code, grb=GRB):
    size_code = getattr(grb, "ERROR_SIZE_LIMIT_EXCEEDED", None)
    return LICENSE_LIMIT if size_code is not None and error_code == size_code else ERROR


def classify_gurobi_status(status, grb=GRB):
    if status == grb.INFEASIBLE:
        return INFEASIBLE
    if status == grb.TIME_LIMIT:
        return TIMEOUT
    return ERROR


def solve(givens: Givens, box_size: int = 3, time_limit_s: float = 60.0) -> BaselineResult:
    n = box_size * box_size
    t0 = time.perf_counter()
    t_build = 0.0
    solve_started = None
    try:
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()
        m = gp.Model(env=env)
        m.Params.Threads = 1
        m.Params.TimeLimit = time_limit_s

        x = m.addVars(n, n, n, vtype=GRB.BINARY, name="x")
        for r in range(n):
            for c in range(n):
                m.addConstr(gp.quicksum(x[r, c, v] for v in range(n)) == 1)
        for v in range(n):
            for r in range(n):
                m.addConstr(gp.quicksum(x[r, c, v] for c in range(n)) == 1)
            for c in range(n):
                m.addConstr(gp.quicksum(x[r, c, v] for r in range(n)) == 1)
            for br in range(0, n, box_size):
                for bc in range(0, n, box_size):
                    m.addConstr(gp.quicksum(
                        x[r, c, v]
                        for r in range(br, br + box_size) for c in range(bc, bc + box_size)) == 1)
        for r, c, v in givens:
            m.addConstr(x[r, c, v] == 1)
        m.setObjective(0, GRB.MINIMIZE)
        m.update()
        t_build = time.perf_counter() - t0

        solve_started = time.perf_counter()
        m.optimize()
        t_solve = time.perf_counter() - solve_started
    except gp.GurobiError as e:
        error_code = getattr(e, "errno", None)
        status = classify_gurobi_error_code(error_code)
        if solve_started is None:
            t_build = time.perf_counter() - t0
            t_solve = 0.0
        else:
            t_solve = time.perf_counter() - solve_started
        return BaselineResult(status, None, t_build, t_solve,
                              diagnostic(NAME, VERSION, f"GurobiError errno={error_code}", str(e)))
    except Exception as e:
        if solve_started is None:
            t_build = time.perf_counter() - t0
            t_solve = 0.0
        else:
            t_solve = time.perf_counter() - solve_started
        return BaselineResult(ERROR, None, t_build, t_solve,
                              diagnostic(NAME, VERSION, "exception", f"{type(e).__name__}: {e}"))

    if m.SolCount > 0:
        grid = [[next(v + 1 for v in range(n) if x[r, c, v].X > 0.5) for c in range(n)] for r in range(n)]
        return BaselineResult(SOLVED, grid, t_build, t_solve)
    classified = classify_gurobi_status(m.Status)
    return BaselineResult(classified, None, t_build, t_solve,
                          diagnostic(NAME, VERSION, f"Gurobi status={m.Status}"))
