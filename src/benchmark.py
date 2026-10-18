"""Benchmark harness for the Sudoku solvers.

Three experiments, all on generated puzzles (see src/puzzle_io.py):

  sat       every AMO encoding x every SAT solver, per grid size
  ablation  with and without the per-unit "at least once" clauses (Weber 2005)
  baseline  ILP / CP solvers (CP-SAT, Gurobi, CPLEX MIP, CPLEX CP, CBC)
  size      encoding size only (variables, clauses, build time), up to 64x64, no solving

Every job runs in its own process so a hard instance can be killed at the time
limit without taking the harness down, and so memory is returned between jobs.
Rows are appended to the CSV as soon as they finish; rerunning with the same
--out skips jobs that are already there.

Timing columns
  t_encode  building the CNF (SAT) or the model (ILP/CP)
  t_solve   SAT: loading the clauses into the solver plus solving; ILP/CP: solving
  t_total   t_encode + t_solve

Alongside the CSV, <out dir>/meta.json records the environment (library versions, CPU,
platform, command-line parameters) of every invocation, which SAT solvers were usable, and
the content hash of every puzzle; puzzles of grids up to 25x25 are also saved as text files in
<out dir>/instances/. Neither file is overwritten: meta.json gains one entry per invocation.

    python -m src.benchmark --boxes 3 4 5 6 --puzzles 3 --out results/results.csv
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import multiprocessing as mp
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone

from src.baselines import BASELINE_MODULES, load_baseline
from src.baselines.common import SOLVED
from src.encodings.amo import AMO_ENCODERS
from src.puzzle_io import generate_solved_grid, givens_to_string, is_valid_solution, puncture
from src.solver import solve
from src.sudoku_encoder import SudokuEncoder

SAT_SOLVER_CANDIDATES = ["cadical195", "glucose42", "kissat404", "minisat22"]
MAX_SAVED_BOX = 5  # the puzzle text format covers grids up to 25x25
PACKAGES = ["python-sat", "ortools", "gurobipy", "cplex", "docplex", "pulp", "matplotlib", "pandas"]
ABLATION_AMOS = ["pairwise", "sequential", "product"]
ABLATION_SOLVER = "cadical195"
FIELDS = ["experiment", "family", "method", "solver", "unit_alo", "box", "n", "puzzle", "keep_ratio",
          "n_vars", "n_clauses", "t_encode", "t_solve", "t_total", "status", "valid", "message"]


def make_puzzle(box, puzzle, keep_ratio):
    seed = box * 1000 + puzzle
    grid = generate_solved_grid(box, seed=seed)
    return puncture(grid, keep_ratio=keep_ratio, seed=seed)


def run_sat(job):
    box, puzzle, keep = job["box"], job["puzzle"], job["keep_ratio"]
    givens = make_puzzle(box, puzzle, keep)
    encoder = SudokuEncoder(box_size=box, amo_method=job["method"], unit_alo=job["unit_alo"])

    t0 = time.perf_counter()
    clauses, n_vars = encoder.encode(givens)
    t_encode = time.perf_counter() - t0

    t0 = time.perf_counter()
    grid = solve(encoder, clauses, solver_name=job["solver"])
    t_solve = time.perf_counter() - t0

    ok = grid is not None and is_valid_solution(grid, givens, box)
    return dict(n_vars=n_vars, n_clauses=len(clauses), t_encode=t_encode, t_solve=t_solve,
                status="solved" if grid is not None else "unsat", valid=ok, message="")


def run_size(job):
    """Encode only. Shows how each AMO encoding scales without paying for a hard solve."""
    box = job["box"]
    givens = make_puzzle(box, job["puzzle"], job["keep_ratio"])
    encoder = SudokuEncoder(box_size=box, amo_method=job["method"], unit_alo=True)
    t0 = time.perf_counter()
    clauses, n_vars = encoder.encode(givens)
    t_encode = time.perf_counter() - t0
    return dict(n_vars=n_vars, n_clauses=len(clauses), t_encode=t_encode, t_solve=0.0,
                status="encoded", valid="", message="")


def run_baseline(job):
    box, puzzle, keep = job["box"], job["puzzle"], job["keep_ratio"]
    givens = make_puzzle(box, puzzle, keep)
    module = load_baseline(job["method"])
    if module is None:
        return dict(status="missing_library", valid="", message="library not installed",
                    t_encode=0.0, t_solve=0.0, n_vars="", n_clauses="")
    result = module.solve(givens, box_size=box, time_limit_s=job["time_limit"])
    ok = result.grid is not None and is_valid_solution(result.grid, givens, box)
    return dict(n_vars="", n_clauses="", t_encode=result.t_build, t_solve=result.t_solve,
                status=result.status, valid=ok if result.status == SOLVED else "", message=result.message)


def _child(conn, job):
    try:
        runner = {"sat": run_sat, "size": run_size}.get(job["family"], run_baseline)
        out = runner(job)
    except MemoryError:
        out = dict(status="memory_error", valid="", message="MemoryError")
    except Exception as e:  # keep the harness alive, record what happened
        out = dict(status="error", valid="", message=f"{type(e).__name__}: {e}"[:300])
    conn.send(out)
    conn.close()


def run_job(job, timeout):
    ctx = mp.get_context("fork")
    parent, child = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_child, args=(child, job))
    proc.start()
    child.close()
    if parent.poll(timeout):
        try:
            out = parent.recv()
        except EOFError:
            out = dict(status="crashed", valid="", message="worker died")
    else:
        out = dict(status="timeout", valid="", message=f">{timeout}s", t_solve=float(timeout))
    proc.terminate()
    proc.join(5)
    if proc.is_alive():
        proc.kill()
        proc.join()
    return out


def job_key(job):
    return (job["experiment"], job["method"], job["solver"], str(job["unit_alo"]), job["box"], job["puzzle"],
            str(job["keep_ratio"]))


def build_jobs(args, sat_solvers, ablation_solver):
    jobs = []
    common = dict(keep_ratio=args.keep_ratio, time_limit=args.timeout)
    for box in args.boxes:
        for puzzle in range(args.puzzles):
            if "sat" in args.only:
                for method in AMO_ENCODERS:
                    for solver in sat_solvers:
                        jobs.append(dict(experiment="sat", family="sat", method=method, solver=solver,
                                         unit_alo=True, box=box, puzzle=puzzle, **common))
            if "ablation" in args.only and ablation_solver:
                for method in ABLATION_AMOS:
                    jobs.append(dict(experiment="ablation", family="sat", method=method,
                                     solver=ablation_solver, unit_alo=False, box=box, puzzle=puzzle, **common))
            if "baseline" in args.only:
                for name in BASELINE_MODULES:
                    jobs.append(dict(experiment="baseline", family="ilp_cp", method=name, solver=name,
                                     unit_alo="", box=box, puzzle=puzzle, **common))
    if "size" in args.only:
        for box in args.size_boxes:
            for method in AMO_ENCODERS:
                if box >= 8 and method == "pairwise":
                    continue  # ~33M clauses as Python lists does not fit in memory
                jobs.append(dict(experiment="size", family="size", method=method, solver="-",
                                 unit_alo=True, box=box, puzzle=0, **common))
    return jobs


def probe_sat_solvers(candidates):
    """Which PySAT solvers can really be built and run here. Returns (usable, {name: reason})."""
    usable, unusable = [], {}
    try:
        from pysat.solvers import Solver
    except Exception as e:
        return [], {name: f"PySAT not importable: {type(e).__name__}: {e}"[:200] for name in candidates}
    for name in candidates:
        try:
            with Solver(name=name, bootstrap_with=[[1, 2], [-1]]) as solver:
                if not solver.solve():
                    raise RuntimeError("probe formula reported UNSAT")
            usable.append(name)
        except Exception as e:
            unusable[name] = f"{type(e).__name__}: {e}"[:200]
    return usable, unusable


def package_versions():
    out = {}
    for name in PACKAGES:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5,
                              cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip() or None
    except Exception:
        return None


def puzzle_hash(givens):
    """SHA-256 (first 12 hex digits) of the canonical clue list, independent of file format."""
    text = ";".join(f"{r},{c},{v}" for r, c, v in sorted(givens))
    return hashlib.sha256(text.encode("ascii")).hexdigest()[:12]


def save_instances(args, out_dir):
    """Describe every benchmark puzzle (with content hash) and save those that fit the text format.
    Refuses to overwrite a saved puzzle whose content differs."""
    records = []
    inst_dir = os.path.join(out_dir, "instances")
    for box in args.boxes:
        for puzzle in range(args.puzzles):
            givens = make_puzzle(box, puzzle, args.keep_ratio)
            record = dict(box=box, n=box * box, puzzle=puzzle, seed=box * 1000 + puzzle,
                          keep_ratio=args.keep_ratio, n_clues=len(givens), sha256_12=puzzle_hash(givens),
                          file=None)
            if box <= MAX_SAVED_BOX:
                os.makedirs(inst_dir, exist_ok=True)
                name = f"b{box}-p{puzzle}-k{args.keep_ratio}-{record['sha256_12']}.suk"
                path = os.path.join(inst_dir, name)
                text = givens_to_string(givens, box) + "\n"
                if os.path.exists(path):
                    with open(path) as f:
                        if f.read() != text:
                            raise SystemExit(f"{path} exists with different content; refusing to overwrite")
                else:
                    with open(path, "w") as f:
                        f.write(text)
                record["file"] = os.path.join("instances", name)
            records.append(record)
    return records


def write_meta(path, args, usable, unusable, ablation_solver, instances):
    """Append one record for this invocation to meta.json (created on first use, never overwritten)."""
    run = dict(
        started_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        command=" ".join([sys.executable, "-m", "src.benchmark"] + sys.argv[1:]),
        parameters=vars(args),
        python=platform.python_version(), platform=platform.platform(), machine=platform.machine(),
        cpu_count=os.cpu_count(), git_commit=git_commit(), packages=package_versions(),
        sat_solvers_used=usable, sat_solvers_unusable=unusable, ablation_solver=ablation_solver,
        instances=instances,
    )
    meta = {"runs": []}
    if os.path.exists(path):
        with open(path) as f:
            meta = json.load(f)
    meta["runs"].append(run)
    with open(path, "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
        f.write("\n")


def load_done(path):
    if not os.path.exists(path):
        return set()
    with open(path, newline="") as f:
        return {(r["experiment"], r["method"], r["solver"], r["unit_alo"], int(r["box"]), int(r["puzzle"]),
                 r["keep_ratio"]) for r in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boxes", type=int, nargs="+", default=[3, 4, 5, 6], help="3->9x9, 4->16x16, 5->25x25 ...")
    ap.add_argument("--puzzles", type=int, default=3, help="puzzles per grid size")
    ap.add_argument("--keep-ratio", type=float, default=0.4,
                    help="fraction of cells given; around 0.3-0.5 is the hard region on 36x36")
    ap.add_argument("--size-boxes", type=int, nargs="+", default=[3, 4, 5, 6, 7, 8],
                    help="grid sizes for the encode-only 'size' experiment")
    ap.add_argument("--timeout", type=int, default=60, help="seconds per job")
    ap.add_argument("--only", nargs="+", default=["sat", "ablation", "baseline"],
                    choices=["sat", "ablation", "baseline", "size"])
    ap.add_argument("--out", default="results/results.csv")
    args = ap.parse_args()

    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)

    usable, unusable = probe_sat_solvers(SAT_SOLVER_CANDIDATES)
    for name, reason in unusable.items():
        print(f"SAT solver {name} skipped: {reason}", flush=True)
    ablation_solver = ABLATION_SOLVER if ABLATION_SOLVER in usable else (usable[0] if usable else None)
    if ablation_solver != ABLATION_SOLVER:
        print(f"ablation solver {ABLATION_SOLVER} unusable; using {ablation_solver}", flush=True)
    write_meta(os.path.join(out_dir, "meta.json"), args, usable, unusable, ablation_solver,
               save_instances(args, out_dir))

    done = load_done(args.out)
    jobs = [j for j in build_jobs(args, usable, ablation_solver) if job_key(j) not in done]
    print(f"{len(jobs)} jobs to run ({len(done)} already in {args.out})", flush=True)

    new_file = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for i, job in enumerate(jobs, 1):
            # Baselines enforce their own limit; the process-level timeout is only a backstop.
            timeout = args.timeout + (30 if job["family"] == "ilp_cp" else 5)
            if job["family"] == "size":
                timeout = 600
            res = run_job(job, timeout)
            t_enc, t_sol = res.get("t_encode", 0.0), res.get("t_solve", 0.0)
            row = dict(experiment=job["experiment"], family=job["family"], method=job["method"],
                       solver=job["solver"], unit_alo=job["unit_alo"], box=job["box"], n=job["box"] ** 2,
                       puzzle=job["puzzle"], keep_ratio=job["keep_ratio"],
                       n_vars=res.get("n_vars", ""), n_clauses=res.get("n_clauses", ""),
                       t_encode=round(t_enc, 6), t_solve=round(t_sol, 6), t_total=round(t_enc + t_sol, 6),
                       status=res.get("status", ""), valid=res.get("valid", ""), message=res.get("message", ""))
            writer.writerow(row)
            f.flush()
            print(f"[{i}/{len(jobs)}] {job['experiment']:8s} n={row['n']:<3} {job['method']:17s} "
                  f"{job['solver']:11s} {row['status']:14s} total={row['t_total']:.3f}s", flush=True)


if __name__ == "__main__":
    main()