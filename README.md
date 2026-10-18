# Sudoku SAT Solver

## Install and run

Python 3 is required. Install the SAT solver dependency and the CP-SAT package
used by the baseline and benchmark:

```bash
python3 -m pip install -r requirements.txt
```

The command-line solver requires PySAT. `ortools` is needed for the CP-SAT
baseline and paired benchmark. To run the correctness tests, install `pytest`
as a development dependency (`python3 -m pip install pytest`).

From the repository root, solve the included example:

```bash
python3 main.py data/puzzles/example_9x9.suk --box 3
```

The command prints encoding statistics followed by the solved grid, or
`UNSATISFIABLE` for an inconsistent puzzle. A missing puzzle path is reported
as a CLI error. `python3 main.py --help` shows the supported CLI options.

Run the project correctness tests with:

```bash
python3 -m pytest src/tests
```

Puzzle files contain exactly `N²` non-whitespace cells in row-major order,
where `N = box_size²`. Use `.` for an empty cell (`0` is accepted as a legacy
empty-cell marker). Values 1–9 are written as `1`–`9`; values 10–35 use
lowercase `a`–`z` (base 36). The supported grid sizes are 1×1, 4×4, 9×9,
16×16, and 25×25; `--box` must be from 1 through 5. Cell values in a
serialized solution use the same symbols, with `.` representing an empty cell.

## Reproducible benchmark

Run the small pilot first, then run the full three-seed matrix if the pilot
confirms the selected sizes and installed solver backends are usable:

```bash
python3 -m src.benchmark --mode pilot
python3 -m src.benchmark --mode full
```

The default matrix uses 9×9 and 16×16 puzzles, generated from fixed seeds and
punctured at a 0.35 keep ratio. A punctured grid is guaranteed to have at least
the generated completion, but is **not guaranteed to have a unique solution**.
Each exact puzzle is saved beside its raw CSV, and every solver/encoding runs
against the same `instance_id`. The CSV records encoding and solve time
separately, status, timeout limit, variable/clause counts, versions, and
observed memory metrics. `n_clauses` is blank for CP-SAT; its model constraint
count is recorded as `n_constraints`.

The benchmark probes PySAT solver availability and uses at most two tested
solvers. It records unavailable backends as rows instead of inventing results.
The pilot defaults to a 10-second solve limit; the full matrix defaults to 30
seconds. Override with `--time-limit SECONDS`. Output paths are timestamped;
an existing CSV or paired-instance directory causes the run to stop rather
than overwrite raw data. `python_peak_mb` tracks Python allocations;
`rss_high_water_mb` is the process high-water RSS and can include earlier
configurations in the same run.

Regenerate the saved benchmark summary and chart without running solvers:

```bash
python3 src/analyze_benchmark.py
```

The generated Markdown report and summary CSV/SVG chart are written to
`results/reports/`. The script reads the existing pilot/full CSVs and does not
change them.

## Data Availability

Source code, sample input, raw benchmark CSVs, paired puzzle instances, and
generated benchmark summaries are available in the project repository:
https://github.com/canknotcaw/sudoku_sat_solver
