"""Benchmark harness: AMO encoding x SAT solver x puzzle size.

Produces results.csv with columns ready to plot (e.g. with cnsplots for
BTL2, or matplotlib/pandas for BTL1): box_size, amo_method, solver, n_vars,
n_clauses, t_encode, t_solve, solved.
"""
import csv
import time

from src.encodings.amo import AMO_ENCODERS
from src.puzzle_io import generate_solved_grid, puncture
from src.solver import solve
from src.sudoku_encoder import SudokuEncoder

BOX_SIZES = [3, 4]            # 9x9, 16x16 -- add 5 (25x25) once this runs comfortably
AMO_METHODS = list(AMO_ENCODERS)
SAT_SOLVERS = ["cadical195", "glucose42"]
TRIALS_PER_CONFIG = 3
KEEP_RATIO = 0.35


def run(out_csv: str = "results.csv"):
    rows = []
    for box in BOX_SIZES:
        base = generate_solved_grid(box, seed=box)
        givens = puncture(base, keep_ratio=KEEP_RATIO, seed=box)

        for amo_method in AMO_METHODS:
            for solver_name in SAT_SOLVERS:
                for trial in range(TRIALS_PER_CONFIG):
                    encoder = SudokuEncoder(box_size=box, amo_method=amo_method)

                    t0 = time.time()
                    clauses, n_vars = encoder.encode(givens)
                    t_encode = time.time() - t0

                    t0 = time.time()
                    grid = solve(encoder, clauses, solver_name=solver_name)
                    t_solve = time.time() - t0

                    row = {
                        "box_size": box,
                        "grid_n": box * box,
                        "amo_method": amo_method,
                        "solver": solver_name,
                        "trial": trial,
                        "n_vars": n_vars,
                        "n_clauses": len(clauses),
                        "t_encode": t_encode,
                        "t_solve": t_solve,
                        "solved": grid is not None,
                    }
                    rows.append(row)
                    print(row)

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {out_csv}")


if __name__ == "__main__":
    run()
