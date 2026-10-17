"""CLI: solve a single Sudoku puzzle file with a chosen AMO encoding and SAT solver."""
import argparse
import time

from src.encodings.amo import AMO_ENCODERS
from src.puzzle_io import grid_to_string, parse_string
from src.solver import solve, solve_check_unique
from src.sudoku_encoder import SudokuEncoder


def main():
    ap = argparse.ArgumentParser(description="SAT-based Sudoku solver (PySAT)")
    ap.add_argument("puzzle_file", help="Path to a puzzle text file (row-major, '.' or '0' = blank)")
    ap.add_argument("--box", type=int, default=3, help="Box size: 3 -> 9x9, 4 -> 16x16, 5 -> 25x25 ...")
    ap.add_argument("--amo", default="pairwise", choices=list(AMO_ENCODERS))
    ap.add_argument("--solver", default="cadical195", help="PySAT solver name")
    ap.add_argument("--check-unique", action="store_true",
                     help="After solving, verify the solution is unique (adds a blocking "
                          "clause and re-solves; prints a second solution if one is found)")
    args = ap.parse_args()

    with open(args.puzzle_file) as f:
        text = f.read()
    givens = parse_string(text, box_size=args.box)

    encoder = SudokuEncoder(box_size=args.box, amo_method=args.amo)

    t0 = time.time()
    clauses, n_vars = encoder.encode(givens)
    t_encode = time.time() - t0

    print(f"# amo={args.amo} solver={args.solver} vars={n_vars} clauses={len(clauses)} "
          f"encode={t_encode:.3f}s", end=" ")

    if not args.check_unique:
        t0 = time.time()
        grid = solve(encoder, clauses, solver_name=args.solver)
        print(f"solve={time.time() - t0:.3f}s")
        print("UNSATISFIABLE" if grid is None else grid_to_string(grid))
        return

    t0 = time.time()
    result = solve_check_unique(encoder, clauses, solver_name=args.solver)
    print(f"solve={time.time() - t0:.3f}s")

    if result.grid is None:
        print("UNSATISFIABLE")
        return

    print(grid_to_string(result.grid))
    if result.is_unique:
        print("# UNIQUE solution")
    else:
        print("# NOT unique -- found a second, different solution:")
        print(grid_to_string(result.second_grid))


if __name__ == "__main__":
    main()
