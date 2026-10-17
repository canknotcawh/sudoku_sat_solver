"""Read/write Sudoku puzzles and generate larger ones for scaling benchmarks."""
import random
from typing import List, Tuple

Givens = List[Tuple[int, int, int]]


def parse_string(text: str, box_size: int = 3) -> Givens:
    """Parse a flattened, row-major puzzle string ('.' or '0' = blank) into
    0-indexed givens (r, c, v). Uses base-36 digits so values above 9 work
    for 16x16 / 25x25 grids (0-9, then a-z)."""
    n = box_size * box_size
    chars = [ch for ch in text if not ch.isspace()]
    if len(chars) != n * n:
        raise ValueError(f"Expected {n * n} cells for box_size={box_size}, got {len(chars)}")

    givens: Givens = []
    for i, ch in enumerate(chars):
        if ch in (".", "0"):
            continue
        r, c = divmod(i, n)
        givens.append((r, c, int(ch, 36) - 1))
    return givens


_DIGITS = "0123456789abcdefghijklmnopqrstuvwxyz"


def grid_to_string(grid: List[List[int]]) -> str:
    """Inverse of parse_string: value v -> _DIGITS[v], matching int(ch, 36) - 1
    used when reading givens, so a cell round-trips through the same char."""
    def enc(v: int) -> str:
        return "." if v == 0 else _DIGITS[v]

    return "\n".join("".join(enc(v) for v in row) for row in grid)


def generate_solved_grid(box_size: int, seed: int = None) -> List[List[int]]:
    """A valid, fully solved grid via the classic diagonal-shift base pattern,
    then randomly relabeled so successive calls don't all look identical."""
    if seed is not None:
        random.seed(seed)
    n = box_size * box_size

    def base(r: int, c: int) -> int:
        return (r * box_size + r // box_size + c) % n

    grid = [[base(r, c) + 1 for c in range(n)] for r in range(n)]

    values = list(range(1, n + 1))
    random.shuffle(values)
    remap = {i + 1: values[i] for i in range(n)}
    return [[remap[v] for v in row] for row in grid]


def puncture(grid: List[List[int]], keep_ratio: float = 0.3, seed: int = None) -> Givens:
    """Keep only `keep_ratio` of the cells as givens (0-indexed values).
    Note: does not guarantee a unique solution — fine for solve-time
    benchmarking, not for generating "real" puzzles to hand-solve."""
    if seed is not None:
        random.seed(seed)
    n = len(grid)
    givens: Givens = []
    for r in range(n):
        for c in range(n):
            if random.random() < keep_ratio:
                givens.append((r, c, grid[r][c] - 1))
    return givens
