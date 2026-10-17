"""Reproducible paired Sudoku SAT versus CP-SAT benchmark."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import platform
from pathlib import Path
import resource
import sys
import threading
import time
import tracemalloc
from typing import Optional, Sequence

from src.encodings.amo import AMO_ENCODERS
from src.puzzle_io import generate_solved_grid, grid_to_string, puncture
from src.sudoku_encoder import SudokuEncoder

BOX_SIZES = (3, 4)
PILOT_SEEDS = (17,)
FULL_SEEDS = (17, 29, 43)
KEEP_RATIO = 0.35
SAT_CANDIDATES = (
    "cadical195", "glucose42", "minisat22", "maplechrono", "maplecm",
    "maplesat", "glucose4", "glucose3", "minicard", "mergesat3",
    "lingeling", "cryptosat", "cadical153", "cadical103",
)
CSV_FIELDS = (
    "run_id", "instance_id", "instance_path", "seed", "box_size", "grid_size",
    "clue_count", "keep_ratio", "amo_method", "solver", "backend", "status",
    "solved", "timeout", "time_limit_s", "n_vars", "n_clauses", "n_constraints",
    "t_encode", "t_solve", "python_peak_mb", "rss_high_water_mb", "version", "error",
)


def _package_version(distribution):
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _probe_sat_solvers(limit=2):
    try:
        from pysat.solvers import Solver
    except Exception as exc:
        return [], f"PySAT unavailable: {type(exc).__name__}: {exc}"
    available = []
    for name in SAT_CANDIDATES:
        try:
            with Solver(name=name) as solver:
                if not (hasattr(solver, "interrupt") and hasattr(solver, "solve_limited")):
                    continue
            available.append(name)
            if len(available) == limit:
                break
        except Exception:
            continue
    return available, ""


def _probe_cp_sat():
    try:
        from ortools.sat.python import cp_model  # noqa: F401
        return True, ""
    except Exception as exc:
        return False, f"OR-Tools CP-SAT unavailable: {type(exc).__name__}: {exc}"


def _version(backend):
    package = "python-sat" if backend == "pysat" else "ortools"
    return f"Python {platform.python_version()}; {package} {_package_version(package)}"


def _memory_metrics():
    py_peak = tracemalloc.get_traced_memory()[1] / (1024 * 1024)
    try:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss /= 1024 * 1024 if sys.platform == "darwin" else 1024
    except (AttributeError, OSError):
        rss = ""
    return f"{py_peak:.6f}", "" if rss == "" else f"{rss:.6f}"


def _make_instances(seeds):
    instances = []
    for box_size in BOX_SIZES:
        for seed in seeds:
            solved = generate_solved_grid(box_size, seed=seed)
            givens = puncture(solved, keep_ratio=KEEP_RATIO, seed=seed + 100_003)
            n = box_size * box_size
            partial = [[0] * n for _ in range(n)]
            for r, c, value in givens:
                partial[r][c] = value + 1
            puzzle = grid_to_string(partial) + "\n"
            digest = hashlib.sha256(puzzle.encode("ascii")).hexdigest()[:12]
            instances.append({
                "box_size": box_size, "grid_size": n, "seed": seed,
                "givens": givens, "clue_count": len(givens),
                "instance_id": f"b{box_size}-seed{seed}-{digest}", "puzzle": puzzle,
            })
    return instances


def _base_row(run_id, instance, puzzle_path, time_limit_s):
    return {
        "run_id": run_id, "instance_id": instance["instance_id"],
        "instance_path": str(puzzle_path), "seed": instance["seed"],
        "box_size": instance["box_size"], "grid_size": instance["grid_size"],
        "clue_count": instance["clue_count"], "keep_ratio": KEEP_RATIO,
        "amo_method": "", "solver": "", "backend": "", "status": "",
        "solved": False, "timeout": False, "time_limit_s": time_limit_s,
        "n_vars": "", "n_clauses": "", "n_constraints": "",
        "t_encode": "", "t_solve": "", "python_peak_mb": "",
        "rss_high_water_mb": "", "version": "", "error": "",
    }


def _is_valid_solution(grid, givens, box_size):
    n = box_size * box_size
    expected = set(range(1, n + 1))
    if grid is None or len(grid) != n or any(len(row) != n or set(row) != expected for row in grid):
        return False
    if any({grid[r][c] for r in range(n)} != expected for c in range(n)):
        return False
    for br in range(0, n, box_size):
        for bc in range(0, n, box_size):
            if {grid[r][c] for r in range(br, br + box_size)
                for c in range(bc, bc + box_size)} != expected:
                return False
    return all(grid[r][c] == value + 1 for r, c, value in givens)


def _run_sat(instance, method, solver_name, time_limit_s):
    from pysat.solvers import Solver

    tracemalloc.reset_peak()
    started = time.perf_counter()
    encoder = SudokuEncoder(instance["box_size"], amo_method=method)
    clauses, n_vars = encoder.encode(instance["givens"])
    t_encode = time.perf_counter() - started

    result = None
    started = time.perf_counter()
    with Solver(name=solver_name, bootstrap_with=clauses) as solver:
        timer = threading.Timer(time_limit_s, solver.interrupt)
        timer.daemon = True
        timer.start()
        try:
            result = solver.solve_limited(expect_interrupt=True)
        finally:
            timer.cancel()
            solver.clear_interrupt()
        grid = encoder.decode(solver.get_model()) if result is True else None
    t_solve = time.perf_counter() - started
    if result is None:
        status, solved, timeout = "TIMEOUT", False, True
    elif result is False:
        status, solved, timeout = "UNSAT", False, False
    elif _is_valid_solution(grid, instance["givens"], instance["box_size"]):
        status, solved, timeout = "SOLVED", True, False
    else:
        status, solved, timeout = "INVALID_SOLUTION", False, False
    py_mem, rss_mem = _memory_metrics()
    return {
        "amo_method": method, "solver": solver_name, "backend": "pysat",
        "status": status, "solved": solved, "timeout": timeout,
        "n_vars": n_vars, "n_clauses": len(clauses), "n_constraints": "",
        "t_encode": f"{t_encode:.9f}", "t_solve": f"{t_solve:.9f}",
        "python_peak_mb": py_mem, "rss_high_water_mb": rss_mem,
        "version": _version("pysat"), "error": "",
    }


def _run_cp_sat(instance, time_limit_s):
    from src.baselines.cp_sat_solver import solve_cp_sat_detailed

    tracemalloc.reset_peak()
    result = solve_cp_sat_detailed(
        instance["givens"], instance["box_size"], time_limit_s=time_limit_s
    )
    status, solved = result.status, False
    if result.grid is not None:
        if _is_valid_solution(result.grid, instance["givens"], instance["box_size"]):
            status, solved = "SOLVED", True
        else:
            status = "INVALID_SOLUTION"
    py_mem, rss_mem = _memory_metrics()
    return {
        "amo_method": "", "solver": "cp-sat", "backend": "ortools-cp-sat",
        "status": status, "solved": solved, "timeout": status == "TIMEOUT",
        "n_vars": result.n_vars, "n_clauses": "", "n_constraints": result.n_constraints,
        "t_encode": f"{result.t_encode:.9f}", "t_solve": f"{result.t_solve:.9f}",
        "python_peak_mb": py_mem, "rss_high_water_mb": rss_mem,
        "version": _version("ortools-cp-sat"), "error": "",
    }


def _unavailable(backend, reason, method=""):
    py_mem, rss_mem = _memory_metrics()
    return {
        "amo_method": method, "solver": "unavailable", "backend": backend,
        "status": "BACKEND_UNAVAILABLE", "solved": False, "timeout": False,
        "version": _version(backend), "error": reason,
        "python_peak_mb": py_mem, "rss_high_water_mb": rss_mem,
    }


def run(mode="full", output=None, time_limit_s=None):
    if mode not in ("pilot", "full"):
        raise ValueError("mode must be 'pilot' or 'full'")
    if time_limit_s is None:
        time_limit_s = 10.0 if mode == "pilot" else 30.0
    if time_limit_s <= 0:
        raise ValueError("time_limit_s must be positive")

    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%SZ") + f"-{mode}"
    output_path = Path(output) if output else Path("results") / f"benchmark_raw_{run_id}.csv"
    instance_path = output_path.parent / f"{output_path.stem}_instances"
    if output_path.exists() or instance_path.exists():
        raise FileExistsError("CSV or paired-instance directory already exists; select a new output path")

    sat_solvers, sat_reason = _probe_sat_solvers(limit=2)
    cp_available, cp_reason = _probe_cp_sat()
    seeds = PILOT_SEEDS if mode == "pilot" else FULL_SEEDS
    instances = _make_instances(seeds)
    print(f"mode={mode}; time_limit_s={time_limit_s}; python={platform.python_version()}")
    print(f"PySAT solvers detected: {sat_solvers or 'none'}; OR-Tools CP-SAT: {cp_available}")
    if sat_reason:
        print(sat_reason)
    elif not sat_solvers:
        print("No candidate solver passed construction and interrupt API checks.")
    if cp_reason:
        print(cp_reason)
    print("Punctured instances are reproducible and share an ID across configurations; uniqueness is not guaranteed.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    instance_path.mkdir(parents=False, exist_ok=False)
    tracemalloc.start()
    rows = 0
    try:
        with output_path.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
            writer.writeheader()
            stream.flush()

            for instance in instances:
                puzzle_file = instance_path / f"{instance['instance_id']}.suk"
                with puzzle_file.open("x", encoding="ascii") as puzzle_stream:
                    puzzle_stream.write(instance["puzzle"])
                print(f"instance={instance['instance_id']} clues={instance['clue_count']}")

                for method in AMO_ENCODERS:
                    for solver_name in (sat_solvers or [None]):
                        try:
                            if solver_name is None:
                                row = _unavailable(
                                    "pysat", sat_reason or "no compatible SAT solver detected", method
                                )
                                tracemalloc.reset_peak()
                                started = time.perf_counter()
                                clauses, n_vars = SudokuEncoder(
                                    instance["box_size"], amo_method=method
                                ).encode(instance["givens"])
                                row.update({
                                    "t_encode": f"{time.perf_counter() - started:.9f}",
                                    "n_vars": n_vars, "n_clauses": len(clauses),
                                })
                                row["python_peak_mb"], row["rss_high_water_mb"] = _memory_metrics()
                            else:
                                row = _run_sat(instance, method, solver_name, time_limit_s)
                        except Exception as exc:
                            row = {
                                "amo_method": method, "solver": solver_name or "unavailable",
                                "backend": "pysat", "status": "ERROR", "solved": False,
                                "timeout": False, "version": _version("pysat"),
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                            row["python_peak_mb"], row["rss_high_water_mb"] = _memory_metrics()
                        record = _base_row(run_id, instance, puzzle_file, time_limit_s)
                        record.update(row)
                        writer.writerow(record)
                        stream.flush()
                        rows += 1
                        print(f"  SAT {method}/{record['solver']}: {record['status']}")

                try:
                    row = (_run_cp_sat(instance, time_limit_s) if cp_available else
                           _unavailable("ortools-cp-sat", cp_reason))
                except Exception as exc:
                    row = {
                        "amo_method": "", "solver": "cp-sat", "backend": "ortools-cp-sat",
                        "status": "ERROR", "solved": False, "timeout": False,
                        "version": _version("ortools-cp-sat"),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    row["python_peak_mb"], row["rss_high_water_mb"] = _memory_metrics()
                record = _base_row(run_id, instance, puzzle_file, time_limit_s)
                record.update(row)
                writer.writerow(record)
                stream.flush()
                rows += 1
                print(f"  CP-SAT: {record['status']}")
    finally:
        tracemalloc.stop()

    print(f"rows={rows}; raw_csv={output_path}; paired_instances={instance_path}")
    if not sat_solvers or not cp_available:
        print("Missing backends are recorded as unavailable; no comparison result is inferred from them.")
    return output_path


def main(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description="Reproducible paired Sudoku SAT/CP-SAT benchmark")
    parser.add_argument("--mode", choices=("pilot", "full"), default="full")
    parser.add_argument("--time-limit", type=float, default=None,
                        help="seconds per solve (pilot default 10; full default 30)")
    parser.add_argument("--output", default=None,
                        help="raw CSV output path (must not exist); puzzle files are saved beside it")
    args = parser.parse_args(argv)
    run(args.mode, args.output, args.time_limit)


if __name__ == "__main__":
    main()
