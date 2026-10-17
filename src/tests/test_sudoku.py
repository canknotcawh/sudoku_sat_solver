import pytest

from src.encodings.amo import AMO_ENCODERS
from src.puzzle_io import generate_solved_grid, grid_to_string, parse_string
from src.sudoku_encoder import SudokuEncoder


EXAMPLE_9X9 = (
    "53..7...." "6..195..." ".98....6." "8...6...3" "4..8.3..1"
    "7...2...6" ".6....28." "...419..5" "....8..79"
)


def _solver_api():
    pytest.importorskip("pysat.solvers")
    from src.solver import solve

    return solve


def _is_valid(grid, givens, box_size):
    n = box_size * box_size
    expected = set(range(1, n + 1))
    if len(grid) != n or any(len(row) != n or set(row) != expected for row in grid):
        return False
    if any({grid[r][c] for r in range(n)} != expected for c in range(n)):
        return False
    for br in range(0, n, box_size):
        for bc in range(0, n, box_size):
            if {grid[r][c] for r in range(br, br + box_size)
                for c in range(bc, bc + box_size)} != expected:
                return False
    return all(grid[r][c] == value + 1 for r, c, value in givens)


def test_puzzle_and_solution_round_trip_for_supported_sizes():
    givens_9 = parse_string(EXAMPLE_9X9, box_size=3)
    partial = [[0] * 9 for _ in range(9)]
    for r, c, value in givens_9:
        partial[r][c] = value + 1
    assert parse_string(grid_to_string(partial), box_size=3) == givens_9

    for box_size in range(1, 6):
        grid = generate_solved_grid(box_size, seed=box_size)
        assert parse_string(grid_to_string(grid), box_size=box_size) == [
            (r, c, grid[r][c] - 1)
            for r in range(len(grid))
            for c in range(len(grid))
        ]


def test_auxiliary_ids_are_disjoint_and_unique_in_sudoku_cnf():
    for name in AMO_ENCODERS:
        encoder = SudokuEncoder(box_size=3, amo_method=name)
        clauses, n_vars = encoder.encode([])
        cell_vars = encoder.n ** 3
        auxiliary_ids = sorted({abs(lit) for clause in clauses for lit in clause if abs(lit) > cell_vars})
        assert all(variable > cell_vars for variable in auxiliary_ids), name
        assert auxiliary_ids == list(range(cell_vars + 1, n_vars + 1)), name


@pytest.mark.parametrize("name", list(AMO_ENCODERS))
def test_valid_9x9_puzzle_solves_with_each_amo(name):
    solve = _solver_api()
    givens = parse_string(EXAMPLE_9X9, box_size=3)
    encoder = SudokuEncoder(box_size=3, amo_method=name)
    clauses, _ = encoder.encode(givens)
    grid = solve(encoder, clauses)
    assert grid is not None
    assert _is_valid(grid, givens, 3)
    assert parse_string(grid_to_string(grid), box_size=3) == [
        (r, c, grid[r][c] - 1) for r in range(9) for c in range(9)
    ]


@pytest.mark.parametrize("name", list(AMO_ENCODERS))
def test_conflicting_givens_are_unsat_with_each_amo(name):
    solve = _solver_api()
    encoder = SudokuEncoder(box_size=3, amo_method=name)
    clauses, _ = encoder.encode([(0, 0, 0), (0, 1, 0)])
    assert solve(encoder, clauses) is None


def test_supported_box_sizes_solve_complete_grids():
    solve = _solver_api()
    for box_size in range(1, 6):
        grid = generate_solved_grid(box_size, seed=box_size)
        givens = [
            (r, c, grid[r][c] - 1)
            for r in range(len(grid))
            for c in range(len(grid))
        ]
        encoder = SudokuEncoder(box_size=box_size, amo_method="sequential")
        clauses, _ = encoder.encode(givens)
        solved = solve(encoder, clauses)
        assert solved == grid
        assert _is_valid(solved, givens, box_size)


def test_out_of_format_dimension_is_rejected():
    with pytest.raises(ValueError, match="grid sizes up to 35"):
        SudokuEncoder(box_size=6)
