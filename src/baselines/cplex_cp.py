"""CPLEX CP Optimizer baseline (docplex.cp), AllDifferent formulation.

The community edition of CP Optimizer rejects models whose search space is
above 2^1000, which in practice allows 9x9 and 16x16 but not 25x25; larger
grids return LICENSE_LIMIT.
"""
import time

from docplex.cp.model import CpoModel
from docplex.cp.utils import CpoException

from .common import (ERROR, INFEASIBLE, LICENSE_LIMIT, SOLVED, BaselineResult, Givens,
                     diagnostic, package_version)

NAME = "cplex_cp"
VERSION = f"docplex={package_version('docplex')},cpoptimizer=not_probed"


def is_observed_cp_license_limit(exc):
    """Match only the specific CP Optimizer 1016 exception recorded in repo results.

    CP Optimizer's docplex exception exposes only a message (no numeric code).
    This requires both the exact observed size-limit sentence and the explicit
    IBM error-1016 reference; broad keyword matching is intentionally avoided.
    """
    if not isinstance(exc, CpoException):
        return False
    text = str(exc)
    return ("Solver error: Problem size limit exceeded." in text and
            "https://ibm.biz/error1016" in text)


def classify_cplex_cp_status(solve_status, fail_status):
    if solve_status == "Infeasible":
        return INFEASIBLE
    # SearchStoppedByLimit is a generic stop-cause (not a time-only cause).
    # Without a specific time-limit result, keep it as ERROR/unknown.
    return ERROR


def solve(givens: Givens, box_size: int = 3, time_limit_s: float = 60.0) -> BaselineResult:
    n = box_size * box_size
    t0 = time.perf_counter()
    t_build = 0.0
    solve_started = None
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

        solve_started = time.perf_counter()
        res = m.solve(TimeLimit=time_limit_s, Workers=1, log_output=None)
        t_solve = time.perf_counter() - solve_started
    except Exception as e:
        msg = str(e)
        status = LICENSE_LIMIT if is_observed_cp_license_limit(e) else ERROR
        if solve_started is None:
            t_build = time.perf_counter() - t0
            t_solve = 0.0
        else:
            t_solve = time.perf_counter() - solve_started
        return BaselineResult(status, None, t_build, t_solve,
                              diagnostic(NAME, VERSION, type(e).__name__, msg))

    if res.is_solution():
        grid = [[res.get_value(cells[r][c]) for c in range(n)] for r in range(n)]
        return BaselineResult(SOLVED, grid, t_build, t_solve)
    status_text = res.get_solve_status()
    if status_text == "Infeasible":
        return BaselineResult(INFEASIBLE, None, t_build, t_solve,
                              diagnostic(NAME, VERSION, f"solve_status={status_text}"))
    stop_cause = res.get_stop_cause() if hasattr(res, "get_stop_cause") else None
    fail_status = res.get_fail_status() if hasattr(res, "get_fail_status") else None
    api_status = f"solve_status={status_text},stop_cause={stop_cause},fail_status={fail_status}"
    classified = classify_cplex_cp_status(status_text, fail_status)
    detail = "termination reason not identified" if classified == ERROR else ""
    return BaselineResult(classified, None, t_build, t_solve,
                          diagnostic(NAME, VERSION, api_status, detail))
