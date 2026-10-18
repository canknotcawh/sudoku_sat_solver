"""End-to-end tests: every AMO encoding must solve Sudoku to a valid grid.

Run:  python3 -m pytest src/tests/test_sudoku.py
"""
import pytest

from src.encodings.amo import AMO_ENCODERS
from src.puzzle_io import generate_solved_grid, parse_string, puncture
from src.solver import solve, solve_check_unique
from src.sudoku_encoder import SudokuEncoder

EXAMPLE_9X9 = (
    "53..7...." "6..195..." ".98....6." "8...6...3" "4..8.3..1"
    "7...2...6" ".6....28." "...419..5" "....8..79"
)


def is_valid(grid, givens, box):
    n = box * box
    want = set(range(1, n + 1))
    rows = [set(r) for r in grid]
    cols = [set(grid[r][c] for r in range(n)) for c in range(n)]
    boxes = [
        set(grid[r][c] for r in range(br, br + box) for c in range(bc, bc + box))
        for br in range(0, n, box) for bc in range(0, n, box)
    ]
    return (
        all(s == want for s in rows + cols + boxes)
        and all(grid[r][c] == v + 1 for r, c, v in givens)
    )


def solve_with(amo, box, givens):
    encoder = SudokuEncoder(box_size=box, amo_method=amo)
    clauses, _ = encoder.encode(givens)
    return solve(encoder, clauses)


def test_example_9x9_all_encodings():
    givens = parse_string(EXAMPLE_9X9, box_size=3)
    grids = []
    for amo in AMO_ENCODERS:
        grid = solve_with(amo, 3, givens)
        assert grid is not None and is_valid(grid, givens, 3), amo
        grids.append(grid)
    # The example puzzle has a unique solution, so all encodings must agree.
    assert all(g == grids[0] for g in grids)


def test_generated_16x16_all_encodings():
    givens = puncture(generate_solved_grid(4, seed=1), keep_ratio=0.35, seed=1)
    for amo in AMO_ENCODERS:
        grid = solve_with(amo, 4, givens)
        assert grid is not None and is_valid(grid, givens, 4), amo


def test_unique_solution_detected_as_unique():
    # The example puzzle is a classic puzzle with a known-unique solution.
    givens = parse_string(EXAMPLE_9X9, box_size=3)
    encoder = SudokuEncoder(box_size=3, amo_method="pairwise")
    clauses, _ = encoder.encode(givens)
    result = solve_check_unique(encoder, clauses)
    assert result.is_unique is True
    assert result.second_grid is None
    assert is_valid(result.grid, givens, 3)


def test_underspecified_puzzle_detected_as_not_unique():
    # A grid with only 2 givens has many solutions -- the two grids returned
    # must both be valid completions, must satisfy the same givens, and must
    # actually differ (otherwise the blocking clause did nothing).
    givens = [(0, 0, 0), (1, 1, 1)]  # cell (0,0)=1, cell (1,1)=2
    encoder = SudokuEncoder(box_size=3, amo_method="pairwise")
    clauses, _ = encoder.encode(givens)
    result = solve_check_unique(encoder, clauses)
    assert result.is_unique is False
    assert is_valid(result.grid, givens, 3)
    assert is_valid(result.second_grid, givens, 3)
    assert result.grid != result.second_grid


def test_uniqueness_agrees_across_encodings():
    # The blocking clause only touches cell variables, so every AMO encoding
    # must reach the same unique/not-unique verdict for the same puzzle.
    givens = parse_string(EXAMPLE_9X9, box_size=3)
    for amo in AMO_ENCODERS:
        encoder = SudokuEncoder(box_size=3, amo_method=amo)
        clauses, _ = encoder.encode(givens)
        result = solve_check_unique(encoder, clauses)
        assert result.is_unique is True, amo


def test_encoder_accepts_large_box_sizes():
    # The text-format limit (box <= 5) must not constrain the encoder: benchmarks use 36x36 and larger.
    for box in (6, 7, 8):
        encoder = SudokuEncoder(box_size=box, amo_method="product")
        assert encoder.n == box * box
    clauses, n_vars = SudokuEncoder(box_size=6, amo_method="sequential").encode([])
    assert n_vars >= 36 ** 3 and clauses


def test_unit_alo_off_gives_fewer_clauses_and_same_solution():
    givens = parse_string(EXAMPLE_9X9, box_size=3)
    with_alo = SudokuEncoder(3, "pairwise", unit_alo=True)
    without = SudokuEncoder(3, "pairwise", unit_alo=False)
    c_with, _ = with_alo.encode(givens)
    c_without, _ = without.encode(givens)
    assert len(c_with) - len(c_without) == 3 * 81      # one ALO per row/column/box and value
    for amo in AMO_ENCODERS:
        encoder = SudokuEncoder(3, amo, unit_alo=False)
        clauses, _ = encoder.encode(givens)
        grid = solve(encoder, clauses)
        assert grid is not None and is_valid(grid, givens, 3), amo
        assert grid == solve(with_alo, c_with)


def test_duplicate_clues_same_value_accepted_different_value_rejected():
    encoder = SudokuEncoder(3)
    once, _ = encoder.encode([(0, 0, 4)])
    twice, _ = encoder.encode([(0, 0, 4), (0, 0, 4)])
    assert once == twice
    with pytest.raises(ValueError, match="conflicting"):
        encoder.encode([(0, 0, 4), (0, 0, 5)])


def test_invalid_clues_rejected():
    encoder = SudokuEncoder(3)
    for bad in ([(0, 0, 9)], [(9, 0, 0)], [(0, -1, 0)], [(0, 0)], [(True, 0, 0)], [(0, 0, 1.0)]):
        with pytest.raises(ValueError):
            encoder.encode(bad)
    for box in (0, -1, True, 2.5):
        with pytest.raises(ValueError):
            SudokuEncoder(box_size=box)
    with pytest.raises(ValueError):
        SudokuEncoder(3, "no-such-amo")


def test_contradictory_clues_in_different_cells_are_unsat():
    encoder = SudokuEncoder(3)
    clauses, _ = encoder.encode([(0, 0, 4), (0, 1, 4)])     # same value twice in a row
    assert solve(encoder, clauses) is None


def test_decode_rejects_invalid_models():
    encoder = SudokuEncoder(3)
    with pytest.raises(ValueError):
        encoder.decode([])                                   # no cell has a value
    good = generate_solved_grid(3, seed=1)
    model = [encoder.var(r, c, good[r][c] - 1) for r in range(9) for c in range(9)]
    assert encoder.decode(model) == good
    with pytest.raises(ValueError):
        encoder.decode(model + [encoder.var(0, 0, (good[0][0]) % 9)])   # a cell with two values
    swapped = [row[:] for row in good]
    swapped[0][0], swapped[0][1] = swapped[0][1], swapped[0][0]
    model = [encoder.var(r, c, swapped[r][c] - 1) for r in range(9) for c in range(9)]
    with pytest.raises(ValueError):
        encoder.decode(model)                                # each cell fine, Sudoku rules broken


if __name__ == "__main__":
    test_example_9x9_all_encodings()
    print("PASS 9x9, all encodings agree")
    test_generated_16x16_all_encodings()
    print("PASS 16x16, all encodings valid")
    test_unique_solution_detected_as_unique()
    print("PASS unique puzzle detected as unique")
    test_underspecified_puzzle_detected_as_not_unique()
    print("PASS underspecified puzzle detected as not unique, second grid differs")
    test_uniqueness_agrees_across_encodings()
    print("PASS uniqueness verdict agrees across all AMO encodings")
    test_encoder_accepts_large_box_sizes()
    test_unit_alo_off_gives_fewer_clauses_and_same_solution()
    test_duplicate_clues_same_value_accepted_different_value_rejected()
    test_invalid_clues_rejected()
    test_contradictory_clues_in_different_cells_are_unsat()
    test_decode_rejects_invalid_models()
    print("PASS encoder validation, unit_alo, large box sizes, decode checks")