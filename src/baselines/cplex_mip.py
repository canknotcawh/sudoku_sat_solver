"""CPLEX MIP baseline (docplex.mp).

Same 0-1 model as the Gurobi baseline. The pip CPLEX is the community edition,
capped at 1000 variables and 1000 constraints, so only 9x9 fits; larger grids
return LICENSE_LIMIT.
"""
import time

from docplex.mp.model import Model
from docplex.mp.utils import DOcplexLimitsExceeded

from .common import (ERROR, INFEASIBLE, LICENSE_LIMIT, SOLVED, TIMEOUT, BaselineResult, Givens,
                     diagnostic, package_version)

NAME = "cplex_mip"
VERSION = f"docplex={package_version('docplex')},cplex={package_version('cplex')}"


def classify_cplex_mip_status_code(status_code):
    # CPLEX MIP status codes: 1016 restricted/community version, 103 infeasible,
    # and 107/108 wall-clock time limit (feasible / no feasible solution).
    if status_code == 1016:
        return LICENSE_LIMIT
    if status_code == 103:
        return INFEASIBLE
    if status_code in (107, 108):
        return TIMEOUT
    return ERROR


def classify_cplex_mip_exception(exc):
    return LICENSE_LIMIT if isinstance(exc, DOcplexLimitsExceeded) else ERROR


def solve(givens: Givens, box_size: int = 3, time_limit_s: float = 60.0) -> BaselineResult:
    n = box_size * box_size
    t0 = time.perf_counter()
    t_build = 0.0
    solve_started = None
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

        solve_started = time.perf_counter()
        sol = m.solve(log_output=False)
        t_solve = time.perf_counter() - solve_started
    except Exception as e:  # classify only the documented/observed size-limit exception
        status = classify_cplex_mip_exception(e)
        if solve_started is None:
            t_build = time.perf_counter() - t0
            t_solve = 0.0
        else:
            t_solve = time.perf_counter() - solve_started
        return BaselineResult(status, None, t_build, t_solve,
                              diagnostic(NAME, VERSION, type(e).__name__, str(e)))

    if sol is not None:
        grid = [[next(v + 1 for v in range(n) if sol.get_value(x[r, c, v]) > 0.5) for c in range(n)]
                for r in range(n)]
        return BaselineResult(SOLVED, grid, t_build, t_solve)
    details = m.get_solve_details()
    status_code = getattr(details, "status_code", None) if details is not None else None
    status_text = getattr(details, "status", "unavailable") if details is not None else "unavailable"
    api_status = f"status_code={status_code},status={status_text}"
    classified = classify_cplex_mip_status_code(status_code)
    return BaselineResult(classified, None, t_build, t_solve,
                          diagnostic(NAME, VERSION, api_status, status_text))
