"""Read/write Sudoku puzzles and generate larger ones for scaling benchmarks."""
import random
import math
from typing import List, Tuple

Givens = List[Tuple[int, int, int]]


def parse_string(text: str, box_size: int = 3) -> Givens:
    """Parse row-major cells into 0-indexed (row, column, value) clues.

    ``.`` is the canonical blank marker; ``0`` is accepted as a legacy blank
    marker. Values 1-9 use ``1``-``9`` and values 10-35 use ``a``-``z``.
    Whitespace between cells is ignored.
    """
    _validate_box_size(box_size)
    n = box_size * box_size
    chars = [ch for ch in text if not ch.isspace()]
    if len(chars) != n * n:
        raise ValueError(f"Expected {n * n} cells for box_size={box_size}, got {len(chars)}")

    givens: Givens = []
    for i, ch in enumerate(chars):
        if ch in (".", "0"):
            continue
        r, c = divmod(i, n)
        try:
            value = int(ch, 36)
        except ValueError as exc:
            raise ValueError(f"Invalid cell symbol {ch!r} at row {r + 1}, column {c + 1}") from exc
        if not 1 <= value <= n:
            raise ValueError(
                f"Cell symbol {ch!r} at row {r + 1}, column {c + 1} represents {value}; "
                f"expected a value from 1 to {n} or a blank marker '.'/'0'"
            )
        givens.append((r, c, value - 1))
    return givens


_DIGITS = "0123456789abcdefghijklmnopqrstuvwxyz"


def _validate_box_size(box_size: int) -> None:
    if isinstance(box_size, bool) or not isinstance(box_size, int) or box_size <= 0:
        raise ValueError("box_size must be a positive integer")
    n = box_size * box_size
    if n > 35:
        raise ValueError("base-36 puzzle format supports grid sizes up to 35 (box_size up to 5)")


def grid_to_string(grid: List[List[int]]) -> str:
    """Serialize a square puzzle grid using ``.`` for blanks and base-36 values.

    Grid values are 1-based (0 means blank); the serialized form parses back
    to 0-based clues.
    """
    if not grid or any(not isinstance(row, list) or len(row) != len(grid) for row in grid):
        raise ValueError("grid must be a non-empty square list of rows")
    n = len(grid)
    box_size = math.isqrt(n)
    if box_size * box_size != n:
        raise ValueError("grid dimension must be a perfect square (box_size squared)")
    _validate_box_size(box_size)

    def enc(v: int) -> str:
        if isinstance(v, bool) or not isinstance(v, int) or not 0 <= v <= n:
            raise ValueError(f"grid values must be integers from 0 to {n}")
        return "." if v == 0 else _DIGITS[v]

    return "\n".join("".join(enc(v) for v in row) for row in grid)


def givens_to_string(givens: Givens, box_size: int) -> str:
    """Serialize 0-indexed givens (row, column, value) in the puzzle text format
    (inverse of parse_string). Subject to the same box_size <= 5 limit."""
    _validate_box_size(box_size)
    n = box_size * box_size
    grid = [[0] * n for _ in range(n)]
    for r, c, v in givens:
        grid[r][c] = v + 1
    return grid_to_string(grid)


def generate_solved_grid(box_size: int, seed: int = None) -> List[List[int]]:
    """A valid, fully solved grid via the classic diagonal-shift base pattern,
    then randomly relabeled so successive calls don't all look identical."""
    rng = random.Random(seed)
    n = box_size * box_size
 
    def base(r: int, c: int) -> int:
        return (r * box_size + r // box_size + c) % n
 
    def shuffled_axis() -> List[int]:
        """A random order of 0..n-1 that keeps every band (group of `box_size`
        rows or columns) together, so the grid stays a valid Sudoku."""
        bands = list(range(box_size))
        rng.shuffle(bands)
        order = []
        for b in bands:
            inside = list(range(box_size))
            rng.shuffle(inside)
            order.extend(b * box_size + i for i in inside)
        return order
 
    rows, cols = shuffled_axis(), shuffled_axis()
    values = list(range(1, n + 1))
    rng.shuffle(values)
    return [[values[base(r, c)] for c in cols] for r in rows]
 
 
def puncture(grid: List[List[int]], keep_ratio: float = 0.3, seed: int = None) -> Givens:
    """Keep only `keep_ratio` of the cells as givens (0-indexed values).
    Note: does not guarantee a unique solution — fine for solve-time
    benchmarking, not for generating "real" puzzles to hand-solve."""
    rng = random.Random(seed)
    n = len(grid)
    givens: Givens = []
    for r in range(n):
        for c in range(n):
            if rng.random() < keep_ratio:
                givens.append((r, c, grid[r][c] - 1))
    return givens


def is_valid_solution(grid: List[List[int]], givens: Givens, box_size: int) -> bool:
    """True iff `grid` is a complete valid Sudoku that keeps every given."""
    n = box_size * box_size
    want = set(range(1, n + 1))
    units = [set(row) for row in grid]
    units += [set(grid[r][c] for r in range(n)) for c in range(n)]
    units += [
        set(grid[r][c] for r in range(br, br + box_size) for c in range(bc, bc + box_size))
        for br in range(0, n, box_size) for bc in range(0, n, box_size)
    ]
    return all(u == want for u in units) and all(grid[r][c] == v + 1 for r, c, v in givens)