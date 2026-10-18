"""Correctness tests for the ILP / CP baselines.

Run:  python -m pytest -q src/tests/test_baselines.py

Every baseline must reproduce the SAT solver's answer on the unique-solution 9x9
puzzle, and the size-limited ones must report license_limit instead of failing when
a model exceeds the free-edition cap. A baseline whose library is not installed is
skipped, not failed.
"""
import pytest

from src.baselines import BASELINE_MODULES, load_baseline
from src.baselines.common import LICENSE_LIMIT, SOLVED
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
