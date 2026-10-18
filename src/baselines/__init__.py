"""ILP / CP baselines. Each module exposes NAME and solve(givens, box_size, time_limit_s).

Imports are lazy so a missing optional library (for example no Gurobi
installed) only disables that one baseline instead of the whole package.
"""
import importlib

BASELINE_MODULES = ["cpsat", "gurobi", "cplex_mip", "cplex_cp", "cbc_ilp"]
_MODULE_FILES = {
    "cpsat": "cp_sat_solver",
    "gurobi": "gurobi_ilp",
    "cplex_mip": "cplex_mip",
    "cplex_cp": "cplex_cp",
    "cbc_ilp": "pulp_cbc",
}


def load_baseline(name: str):
    """Return the baseline module for `name`, or None if its library is not installed."""
    try:
        return importlib.import_module(f"{__name__}.{_MODULE_FILES[name]}")
    except ImportError:
        return None