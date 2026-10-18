"""Tests for puzzle text I/O, the instance generator and the solution checker."""
import pytest

from src.puzzle_io import (generate_solved_grid, givens_to_string, grid_to_string,
                           is_valid_solution, parse_string, puncture)


@pytest.mark.parametrize("box", [3, 4, 5])
def test_solution_round_trip(box):
    n = box * box
    grid = generate_solved_grid(box, seed=7)
    assert parse_string(grid_to_string(grid), box) == [(r, c, grid[r][c] - 1) for r in range(n) for c in range(n)]


@pytest.mark.parametrize("box", [3, 4, 5])
def test_puzzle_round_trip(box):
    givens = puncture(generate_solved_grid(box, seed=3), keep_ratio=0.4, seed=3)
    assert parse_string(givens_to_string(givens, box), box) == givens


def test_generator_is_deterministic_and_seed_sensitive():
    assert generate_solved_grid(4, seed=1) == generate_solved_grid(4, seed=1)
    assert generate_solved_grid(4, seed=1) != generate_solved_grid(4, seed=2)
    assert puncture(generate_solved_grid(4, seed=1), 0.4, seed=5) == puncture(generate_solved_grid(4, seed=1), 0.4, seed=5)


def test_generator_does_not_touch_global_random():
    import random
    random.seed(123)
    expected = random.random()
    random.seed(123)
    generate_solved_grid(4, seed=9)
    puncture(generate_solved_grid(4, seed=9), 0.4, seed=9)
    assert random.random() == expected


@pytest.mark.parametrize("box", [2, 3, 4, 5, 6, 7])
def test_generated_grid_is_valid(box):
    assert is_valid_solution(generate_solved_grid(box, seed=11), [], box)


def test_different_seeds_differ_in_structure_not_only_labels():
    # Relabelling symbols keeps the pattern of equal cells; row/column shuffles do not.
    def pattern(g):
        return [[g[r][c] == g[0][0] for c in range(len(g))] for r in range(len(g))]
    assert len({str(pattern(generate_solved_grid(4, seed=s))) for s in range(8)}) > 1


def test_is_valid_solution_rejects_bad_grids():
    grid = generate_solved_grid(3, seed=1)
    assert is_valid_solution(grid, [(0, 0, grid[0][0] - 1)], 3)
    assert not is_valid_solution(grid, [(0, 0, grid[0][0] % 9)], 3)       # wrong clue
    bad = [row[:] for row in grid]
    bad[0][0], bad[0][1] = bad[0][1], bad[0][0]                            # breaks columns/boxes
    assert not is_valid_solution(bad, [], 3)
    bad = [row[:] for row in grid]
    bad[4][4] = 0
    assert not is_valid_solution(bad, [], 3)


@pytest.mark.parametrize("bad_grid", [
    None,
    [],
    [[1] * 8 for _ in range(9)],
    [[1] * 9 for _ in range(8)],
    [[1] * 9 for _ in range(8)] + [[1] * 8],
    [[1] * 8 + [True]] + [[1] * 9 for _ in range(8)],
    [[1] * 8 + [1.0]] + [[1] * 9 for _ in range(8)],
])
def test_validator_returns_false_for_invalid_grid_shape_or_types(bad_grid):
    assert is_valid_solution(bad_grid, [], 3) is False


@pytest.mark.parametrize("box", [0, -1, True, 2.5, None])
def test_validator_returns_false_for_invalid_box_size(box):
    assert is_valid_solution([], [], box) is False


@pytest.mark.parametrize("clues", [
    None,
    [(0, 0)],
    [(True, 0, 0)],
    [(0, 0, False)],
    [(-1, 0, 0)],
    [(0, 9, 0)],
    [(0, 0, -1)],
    [(0, 0, 9)],
    [(0.0, 0, 0)],
])
def test_validator_returns_false_for_malformed_or_out_of_range_clues(clues):
    grid = generate_solved_grid(3, seed=1)
    assert is_valid_solution(grid, clues, 3) is False


def test_parse_errors_and_format_limit():
    for text, box in (("x" * 81, 3), ("." * 80, 3), ("g" + "." * 80, 3), ("." * 81, 0), ("." * 1296, 6)):
        with pytest.raises(ValueError):
            parse_string(text, box)
    with pytest.raises(ValueError):
        grid_to_string(generate_solved_grid(6, seed=1))
