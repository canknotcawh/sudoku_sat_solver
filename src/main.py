"""CLI: solve a single Sudoku puzzle file with a chosen AMO encoding and SAT solver."""
import argparse
import time
from pathlib import Path

from src.encodings.amo import AMO_ENCODERS
from src.puzzle_io import grid_to_string, parse_string
from src.sudoku_encoder import SudokuEncoder


def main():
    ap = argparse.ArgumentParser(description="SAT-based Sudoku solver (PySAT)")
    ap.add_argument("puzzle_file", help="Row-major puzzle text; '.' is blank ('0' is also accepted)")
    ap.add_argument("--box", type=int, default=3,
                     help="Box size: 3 -> 9x9, 4 -> 16x16, 5 -> 25x25 (maximum supported)")
    ap.add_argument("--amo", default="pairwise", choices=list(AMO_ENCODERS))
    ap.add_argument("--solver", default="cadical195", help="PySAT solver name")
    ap.add_argument("--no-unit-alo", action="store_true",
                     help="Omit the redundant 'value appears at least once' clauses for rows, columns "
                          "and boxes (the per-cell ALO is always kept)")
    ap.add_argument("--check-unique", action="store_true",
                     help="After solving, verify the solution is unique (adds a blocking "
                          "clause and re-solves; prints a second solution if one is found)")
    args = ap.parse_args()

    try:
        text = Path(args.puzzle_file).read_text(encoding="utf-8")
    except OSError as exc:
        ap.error(f"cannot read puzzle file {args.puzzle_file!r}: {exc.strerror or exc}")
    t0 = time.time()
    try:
        givens = parse_string(text, box_size=args.box)
        encoder = SudokuEncoder(box_size=args.box, amo_method=args.amo, unit_alo=not args.no_unit_alo)
        clauses, n_vars = encoder.encode(givens)
    except ValueError as exc:
        ap.error(str(exc))
    t_encode = time.time() - t0

    try:
        from src.solver import solve, solve_check_unique
    except ModuleNotFoundError as exc:
        if exc.name == "pysat":
            ap.error("PySAT is not installed; install project dependencies with `python3 -m pip install -r requirements.txt`")
        raise

    print(f"# amo={args.amo} solver={args.solver} vars={n_vars} clauses={len(clauses)} "
          f"encode={t_encode:.3f}s")

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