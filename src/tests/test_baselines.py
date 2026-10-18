"""Correctness tests for the ILP / CP baselines.

Run:  python -m pytest -q src/tests/test_baselines.py

Every baseline must reproduce the SAT solver's answer on the unique-solution 9x9
puzzle, and the size-limited ones must report license_limit instead of failing when
a model exceeds the free-edition cap. A baseline whose library is not installed is
skipped, not failed.
"""
import pytest
import shutil

from src.baselines import BASELINE_MODULES, load_baseline
from src.baselines.common import ERROR, INFEASIBLE, LICENSE_LIMIT, SOLVED, TIMEOUT, diagnostic
from src.puzzle_io import generate_solved_grid, parse_string, puncture
from src.solver import solve
from src.sudoku_encoder import SudokuEncoder
from src.tests.test_sudoku import EXAMPLE_9X9, is_valid

REQUIRED_LIBRARIES = {
    "cpsat": ["ortools"],
    "gurobi": ["gurobipy"],
    "cplex_mip": ["docplex", "cplex"],
    "cplex_cp": ["docplex", "cplex"],
    "cbc_ilp": ["pulp"],
}


def baseline_or_skip(name):
    for library in REQUIRED_LIBRARIES[name]:
        pytest.importorskip(library)
    if name == "cplex_cp" and shutil.which("cpoptimizer") is None:
        pytest.skip("cplex_cp: CP Optimizer executable is not installed/on PATH")
    module = load_baseline(name)
    if module is None:
        pytest.skip(f"{name}: library not installed")
    return module


def sat_reference(givens, box):
    encoder = SudokuEncoder(box_size=box, amo_method="pairwise")
    clauses, _ = encoder.encode(givens)
    return solve(encoder, clauses)


@pytest.mark.parametrize("name", BASELINE_MODULES)
def test_baseline_matches_sat_on_9x9(name):
    module = baseline_or_skip(name)
    givens = parse_string(EXAMPLE_9X9, box_size=3)
    result = module.solve(givens, box_size=3, time_limit_s=60)
    assert result.status == SOLVED, (name, result.status, result.message)
    assert is_valid(result.grid, givens, 3), name
    assert result.grid == sat_reference(givens, 3), name  # unique solution, so all must agree


@pytest.mark.parametrize("name", BASELINE_MODULES)
def test_16x16_valid_or_license_limit(name):
    module = baseline_or_skip(name)
    givens = puncture(generate_solved_grid(4, seed=1), keep_ratio=0.35, seed=1)
    result = module.solve(givens, box_size=4, time_limit_s=120)
    assert result.status in (SOLVED, LICENSE_LIMIT), (name, result.status, result.message)
    if result.status == SOLVED:
        assert is_valid(result.grid, givens, 4), name


@pytest.mark.parametrize("name", ["gurobi", "cplex_mip"])
def test_size_limited_editions_report_license_limit(name):
    # Holds for the pip-installed free editions. With a full licence this model fits and the test is moot.
    module = baseline_or_skip(name)
    givens = puncture(generate_solved_grid(4, seed=1), keep_ratio=0.35, seed=1)
    result = module.solve(givens, box_size=4, time_limit_s=60)
    if result.status == SOLVED:
        pytest.skip(f"{name}: full licence detected, no size cap")
    assert result.status == LICENSE_LIMIT, (name, result.status, result.message)


def test_cpsat_uses_a_single_worker():
    pytest.importorskip("ortools")
    import inspect
    from src.baselines import cp_sat_solver
    assert "num_workers = 1" in inspect.getsource(cp_sat_solver)


def test_cpsat_status_classification_keeps_unknown_out_of_timeout():
    pytest.importorskip("ortools")
    from ortools.sat.python import cp_model
    from src.baselines.cp_sat_solver import classify_cp_sat_status

    assert classify_cp_sat_status(cp_model.INFEASIBLE, cp_model) == INFEASIBLE
    assert classify_cp_sat_status(cp_model.OPTIMAL, cp_model) == SOLVED
    # UNKNOWN can reflect time, memory, interruption, etc.; it is not a timeout proof.
    assert classify_cp_sat_status(cp_model.UNKNOWN, cp_model) == ERROR


def test_gurobi_uses_solver_error_code_not_message_keywords():
    pytest.importorskip("gurobipy")
    from types import SimpleNamespace
    from src.baselines.gurobi_ilp import classify_gurobi_error_code, classify_gurobi_status

    api = SimpleNamespace(ERROR_SIZE_LIMIT_EXCEEDED=10010)
    assert classify_gurobi_error_code(10010, api) == LICENSE_LIMIT
    assert classify_gurobi_error_code(10009, api) == ERROR
    assert classify_gurobi_error_code(None, api) == ERROR
    statuses = SimpleNamespace(INFEASIBLE=3, TIME_LIMIT=9)
    assert classify_gurobi_status(3, statuses) == INFEASIBLE
    assert classify_gurobi_status(9, statuses) == TIMEOUT
    assert classify_gurobi_status(100, statuses) == ERROR


def test_cplex_mip_uses_status_codes_and_specific_docplex_exception():
    pytest.importorskip("docplex")
    from docplex.mp.utils import DOcplexLimitsExceeded
    from src.baselines.cplex_mip import classify_cplex_mip_exception, classify_cplex_mip_status_code

    assert classify_cplex_mip_status_code(1016) == LICENSE_LIMIT
    assert classify_cplex_mip_status_code(103) == INFEASIBLE
    assert classify_cplex_mip_status_code(107) == TIMEOUT
    assert classify_cplex_mip_status_code(108) == TIMEOUT
    assert classify_cplex_mip_status_code(9999) == ERROR
    observed = DOcplexLimitsExceeded(1200, 1200)
    assert "CPLEX code=1016" in str(observed)
    assert classify_cplex_mip_exception(observed) == LICENSE_LIMIT
    assert classify_cplex_mip_exception(RuntimeError("license size 1016")) == ERROR


def test_cplex_cp_requires_the_observed_typed_1016_diagnostic():
    pytest.importorskip("docplex")
    from docplex.cp.utils import CpoException
    from src.baselines.cplex_cp import is_observed_cp_license_limit, classify_cplex_cp_status

    observed = CpoException(
        "Solver error: Problem size limit exceeded.\n"
        "CP Optimizer Community Edition solves problems with search spaces up to 2^1000.\n"
        "Unrestricted version options (including academia) at https://ibm.biz/error1016\n"
    )
    assert is_observed_cp_license_limit(observed)
    assert not is_observed_cp_license_limit(RuntimeError(str(observed)))
    assert not is_observed_cp_license_limit(CpoException("Problem size limit exceeded"))
    assert classify_cplex_cp_status("Infeasible", "anything") == INFEASIBLE
    assert classify_cplex_cp_status("Unknown", "SearchStoppedByLimit") == ERROR
    assert classify_cplex_cp_status("Unknown", "Unknown") == ERROR


def test_cbc_does_not_infer_timeout_from_not_solved_status():
    pytest.importorskip("pulp")
    from src.baselines.pulp_cbc import classify_pulp_cbc_status

    assert classify_pulp_cbc_status("Infeasible") == INFEASIBLE
    assert classify_pulp_cbc_status("Optimal") == SOLVED
    assert classify_pulp_cbc_status("Not Solved") == ERROR


def test_diagnostic_preserves_solver_version_status_and_original_message():
    message = diagnostic("cplex_mip", "docplex=2.x,cplex=22.x", "status_code=1016", "original detail")
    assert "solver=cplex_mip" in message
    assert "version=docplex=2.x,cplex=22.x" in message
    assert "api_status=status_code=1016" in message
    assert "original=original detail" in message


def test_unavailable_optional_solver_import_still_returns_none(monkeypatch):
    import src.baselines as baselines

    monkeypatch.setattr(baselines.importlib, "import_module",
                        lambda _name: (_ for _ in ()).throw(ImportError("optional dependency absent")))
    assert baselines.load_baseline("gurobi") is None
