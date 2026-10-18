"""End-to-end tests: every AMO encoding must solve Sudoku to a valid grid.

Run:  python3 -m pytest src/tests/test_sudoku.py
"""
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
