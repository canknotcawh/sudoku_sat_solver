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

Timing columns (all seconds)
  t_encode  build the CNF/model inside the worker
  t_solve   SAT solver setup/loading + solve; baseline solver-call duration
  t_total   t_encode + t_solve when those phases are returned by the worker
  t_wall    parent-measured worker wall time, including process startup/IPC

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
import uuid
from datetime import datetime, timezone

from src.baselines import BASELINE_MODULES, load_baseline
from src.baselines.common import SOLVED
from src.encodings.amo import AMO_ENCODERS
from src.puzzle_io import generate_solved_grid, givens_to_string, is_valid_solution, puncture
from src.solver import solve, solve_check_unique
from src.sudoku_encoder import SudokuEncoder
from src.benchmark_schema import FIELDS, LEGACY_FIELDS, V2_FIELDS, V3_FIELDS, SCHEMA_VERSION

SAT_SOLVER_CANDIDATES = ["cadical195", "glucose42", "kissat404", "minisat22"]
MAX_SAVED_BOX = 5  # the puzzle text format covers grids up to 25x25
PACKAGES = ["python-sat", "ortools", "gurobipy", "cplex", "docplex", "pulp", "matplotlib", "pandas"]
ABLATION_AMOS = ["pairwise", "sequential", "product"]
ABLATION_SOLVER = "cadical195"
SEED_STRATEGY = "explicit_seed_or_box_times_1000_plus_index_v2"


def make_puzzle(box, puzzle, keep_ratio, seed=None):
    seed = box * 1000 + puzzle if seed is None else seed
    grid = generate_solved_grid(box, seed=seed)
    return puncture(grid, keep_ratio=keep_ratio, seed=seed)


def run_sat(job):
    box, givens = job["box"], job["givens"]
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
    givens = job["givens"]
    encoder = SudokuEncoder(box_size=box, amo_method=job["method"], unit_alo=True)
    t0 = time.perf_counter()
    clauses, n_vars = encoder.encode(givens)
    t_encode = time.perf_counter() - t0
    return dict(n_vars=n_vars, n_clauses=len(clauses), t_encode=t_encode, t_solve=0.0,
                status="encoded", valid="", message="")


def run_baseline(job):
    box, givens = job["box"], job["givens"]
    module = load_baseline(job["method"])
    if module is None:
        return dict(status="missing_library", valid="", message="library not installed",
                    t_encode="", t_solve="", n_vars="", n_clauses="")
    result = module.solve(givens, box_size=box, time_limit_s=job["time_limit"])
    ok = result.grid is not None and is_valid_solution(result.grid, givens, box)
    return dict(n_vars="", n_clauses="", t_encode=result.t_build, t_solve=result.t_solve,
                status=result.status, valid=ok if result.status == SOLVED else "", message=result.message)


def _child(conn, job):
    try:
        runner = {"sat": run_sat, "size": run_size}.get(job["family"], run_baseline)
        out = runner(job)
    except MemoryError:
        out = dict(status="memory_error", valid="", message="MemoryError", timeout_kind="worker_error")
    except Exception as e:  # keep the harness alive, record what happened
        out = dict(status="error", valid="", timeout_kind="worker_error",
                   message=f"{type(e).__name__}: {e}"[:300])
    conn.send(out)
    conn.close()


def run_job(job, timeout):
    """Run one worker and classify returned solver timeouts vs parent process kills/crashes.

    The returned ``t_wall`` is measured outside the child. A process deadline has no
    reliable phase measurements, so it deliberately returns no t_encode/t_solve.
    """
    ctx = mp.get_context("fork")
    parent, child = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_child, args=(child, job))
    started = time.perf_counter()
    try:
        proc.start()
    except Exception as exc:
        return dict(status="crashed", timeout_kind="worker_crash", valid="",
                    message=f"worker start failed: {type(exc).__name__}: {exc}"[:300],
                    t_wall=time.perf_counter() - started)
    child.close()
    deadline = started + timeout
    out = None
    while out is None:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            if parent.poll(0):
                try:
                    out = parent.recv()
                except EOFError:
                    out = dict(status="crashed", timeout_kind="worker_crash", valid="", message="worker died")
            elif not proc.is_alive():
                out = dict(status="crashed", timeout_kind="worker_crash", valid="", message="worker exited without a result")
            else:
                out = dict(status="timeout", timeout_kind="process", valid="", message=f"worker exceeded {timeout}s")
            break
        if parent.poll(min(remaining, 0.05)):
            try:
                out = parent.recv()
            except EOFError:
                out = dict(status="crashed", timeout_kind="worker_crash", valid="", message="worker died")
            break
        if not proc.is_alive():
            if parent.poll(0):
                continue
            out = dict(status="crashed", timeout_kind="worker_crash", valid="", message="worker exited without a result")
            break
    elapsed = time.perf_counter() - started
    if out.get("status") == "timeout" and out.get("timeout_kind") not in {"solver", "process"}:
        # A baseline can report its own solver-level timeout; SAT has only the outer process deadline.
        out["timeout_kind"] = "solver" if job.get("solver_timeout_s") is not None else "process"
    if out.get("status") == "crashed":
        out.setdefault("timeout_kind", "worker_crash")
    if out.get("status") in {"error", "memory_error"}:
        out.setdefault("timeout_kind", "worker_error")
    out.setdefault("timeout_kind", "none")
    out["t_wall"] = elapsed
    proc.terminate()
    proc.join(5)
    if proc.is_alive():
        proc.kill()
        proc.join()
    return out


def job_key(job):
    return (job["experiment"], job["method"], job["solver"], str(job["unit_alo"]), job["box"], job["puzzle"],
            str(job["keep_ratio"]), job["time_limit"], job["worker_timeout"], job.get("solver_timeout_s"),
            job.get("process_timeout_s"), job.get("seed"),
            job.get("puzzle_sha256"), job.get("split"))


def ratios_for(args):
    return list(getattr(args, "keep_ratios", [getattr(args, "keep_ratio", 0.4)]))


def seeds_for(args, box):
    explicit = getattr(args, "seeds", None)
    return list(explicit) if explicit else [box * 1000 + i for i in range(args.puzzles)]


def build_jobs(args, sat_solvers, ablation_solver):
    jobs = []
    ratios = ratios_for(args)
    tune_count = getattr(args, "tune_count", 0)
    for box in args.boxes:
        for puzzle, seed in enumerate(seeds_for(args, box)):
            for ratio in ratios:
                split = "tune" if puzzle < tune_count else "evaluation"
                common = dict(keep_ratio=ratio, time_limit=args.timeout, seed=seed, split=split)
                givens = make_puzzle(box, puzzle, ratio, seed=seed)
                if "sat" in args.only:
                    for method in AMO_ENCODERS:
                        for solver in sat_solvers:
                            jobs.append(dict(experiment="sat", family="sat", method=method, solver=solver,
                                             unit_alo=True, box=box, puzzle=puzzle, givens=givens, **common))
                if "ablation" in args.only and ablation_solver:
                    for method in ABLATION_AMOS:
                        jobs.append(dict(experiment="ablation", family="sat", method=method,
                                         solver=ablation_solver, unit_alo=False, box=box, puzzle=puzzle,
                                         givens=givens, **common))
                if "baseline" in args.only:
                    for name in BASELINE_MODULES:
                        jobs.append(dict(experiment="baseline", family="ilp_cp", method=name, solver=name,
                                         unit_alo="", box=box, puzzle=puzzle, givens=givens, **common))
    if "size" in args.only:
        for box in args.size_boxes:
            seed = seeds_for(args, box)[0]
            for ratio in ratios:
                givens = make_puzzle(box, 0, ratio, seed=seed)
                for method in AMO_ENCODERS:
                    if box >= 8 and method == "pairwise":
                        continue  # ~33M clauses as Python lists does not fit in memory
                    jobs.append(dict(experiment="size", family="size", method=method, solver="-",
                                     unit_alo=True, box=box, puzzle=0, seed=seed, split="descriptive",
                                     keep_ratio=ratio, time_limit=args.timeout, givens=givens))
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


def source_fingerprint():
    """Hash tracked implementation/dependency inputs, including uncommitted edits."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        listed = subprocess.run(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            capture_output=True, check=True, cwd=root, timeout=5,
        ).stdout.split(b"\0")
    except Exception:
        return None
    digest = hashlib.sha256()
    for raw_path in sorted(p for p in listed if p):
        rel_path = raw_path.decode("utf-8")
        if not (rel_path.endswith(".py") or rel_path in {"requirements.txt", "requirements-extra.txt", "tests.yml"}):
            continue
        path = os.path.join(root, rel_path)
        try:
            with open(path, "rb") as stream:
                data = stream.read()
        except OSError:
            continue
        digest.update(rel_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(data).digest())
    return digest.hexdigest()


def puzzle_hash(givens):
    """SHA-256 (first 12 hex digits) of the canonical clue list, independent of file format."""
    text = ";".join(f"{r},{c},{v}" for r, c, v in sorted(givens))
    return hashlib.sha256(text.encode("ascii")).hexdigest()[:12]


def puzzle_sha256(givens):
    """Full SHA-256 of the canonical clue list for per-row provenance."""
    text = ";".join(f"{r},{c},{v}" for r, c, v in sorted(givens))
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def save_instances(args, out_dir):
    """Describe every benchmark puzzle (with content hash) and save those that fit the text format.
    Refuses to overwrite a saved puzzle whose content differs."""
    records = []
    inst_dir = os.path.join(out_dir, "instances")
    ratios = ratios_for(args)
    requested = set()
    for box in args.boxes:
        for puzzle, seed in enumerate(seeds_for(args, box)):
            for ratio in ratios:
                requested.add((box, puzzle, seed, ratio))
    if "size" in args.only:
        for box in args.size_boxes:
            for ratio in ratios:
                requested.add((box, 0, seeds_for(args, box)[0], ratio))
    for box, puzzle, seed, ratio in sorted(requested):
        givens = make_puzzle(box, puzzle, ratio, seed=seed)
        uniqueness_checked = bool(getattr(args, "check_uniqueness", False))
        uniqueness_status = "not_checked"
        if uniqueness_checked:
            encoder = SudokuEncoder(box_size=box, amo_method="product", unit_alo=True)
            clauses, _ = encoder.encode(givens)
            result = solve_check_unique(encoder, clauses, solver_name=getattr(args, "uniqueness_solver", "cadical195"))
            if result.grid is None:
                raise RuntimeError(f"generated puzzle is unexpectedly UNSAT: box={box}, seed={seed}, ratio={ratio}")
            uniqueness_status = "unique" if result.is_unique else "multiple"
        record = dict(box=box, n=box * box, puzzle=puzzle, seed=seed, keep_ratio=ratio,
                      split="tune" if puzzle < getattr(args, "tune_count", 0) else "evaluation",
                      n_clues=len(givens), sha256_12=puzzle_hash(givens), sha256=puzzle_sha256(givens),
                      uniqueness_checked=uniqueness_checked, uniqueness_status=uniqueness_status,
                      solution_guarantee="generated_completion", file=None)
        if box <= MAX_SAVED_BOX:
            os.makedirs(inst_dir, exist_ok=True)
            name = f"b{box}-p{puzzle}-s{seed}-k{ratio}-{record['sha256_12']}.suk"
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


def canonical_config(args, usable, unusable, ablation_solver):
    """Stable payload for identifying runs whose results are safe to resume."""
    return {
        "schema_version": SCHEMA_VERSION,
        "parameters": {k: v for k, v in vars(args).items() if k not in {"out", "new_run"}},
        "seed_strategy": SEED_STRATEGY,
        "git_commit": git_commit(),
        "source_fingerprint": source_fingerprint(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "packages": package_versions(),
        "sat_solvers_used": usable,
        "sat_solvers_unusable": unusable,
        "ablation_solver": ablation_solver,
    }


def config_hash(config):
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_meta(path, args, usable, unusable, ablation_solver, instances, run_id, cfg_hash):
    """Append one record for this invocation to meta.json (created on first use, never overwritten)."""
    cfg = canonical_config(args, usable, unusable, ablation_solver)
    run = dict(
        schema_version=SCHEMA_VERSION, run_id=run_id, config_hash=cfg_hash,
        seed_strategy=SEED_STRATEGY, config=cfg,
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
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp_path, path)


def resolve_csv_path(path):
    """Keep a legacy CSV byte-for-byte intact and direct new-schema rows to a sidecar."""
    if not os.path.exists(path):
        return path, False
    with open(path, newline="", encoding="utf-8") as f:
        header = next(csv.reader(f), [])
    if header == FIELDS:
        return path, False
    if header in (LEGACY_FIELDS, V2_FIELDS, V3_FIELDS):
        root, ext = os.path.splitext(path)
        sidecar = f"{root}.v4{ext or '.csv'}"
        if os.path.exists(sidecar):
            with open(sidecar, newline="", encoding="utf-8") as f:
                if next(csv.reader(f), []) != FIELDS:
                    raise SystemExit(f"Existing sidecar {sidecar} has an unknown schema; refusing to append.")
        return sidecar, True
    raise SystemExit(f"CSV {path} has an unrecognized schema; refusing to modify it.")


def load_done(path, run_id, cfg_hash):
    if not os.path.exists(path):
        return set()
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != FIELDS:
            return set()  # old schemas are preserved, never treated as completed v3 jobs.
        return {(r["experiment"], r["method"], r["solver"], r["unit_alo"], int(r["box"]), int(r["puzzle"]),
                 r["keep_ratio"], int(r["timeout_s"]), int(r["worker_timeout_s"]),
                 (float(r["solver_timeout_s"]) if r["solver_timeout_s"] else None),
                 float(r["process_timeout_s"]), int(r["seed"]),
                 r["puzzle_sha256"], r["split"])
                for r in reader if r["run_id"] == run_id and r["config_hash"] == cfg_hash}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boxes", type=int, nargs="+", default=[3, 4, 5, 6], help="3->9x9, 4->16x16, 5->25x25 ...")
    ap.add_argument("--puzzles", type=int, default=3, help="puzzles per grid size")
    ap.add_argument("--seeds", type=int, nargs="+", help="explicit puzzle seeds, reused across grid sizes")
    ap.add_argument("--tune-count", type=int, default=0,
                    help="first K seeds form the tuning split; remaining seeds are held-out evaluation")
    ap.add_argument("--keep-ratio", type=float, dest="legacy_keep_ratio",
                    help="deprecated single-density alias for --keep-ratios")
    ap.add_argument("--keep-ratios", type=float, nargs="+", help="clue fractions, e.g. --keep-ratios 0.35 0.45")
    ap.add_argument("--check-uniqueness", action="store_true",
                    help="check each generated puzzle once and record unique/multiple (may be expensive)")
    ap.add_argument("--uniqueness-solver", default="cadical195", help="PySAT solver used for optional uniqueness checks")
    ap.add_argument("--size-boxes", type=int, nargs="+", default=[3, 4, 5, 6, 7, 8],
                    help="grid sizes for the encode-only 'size' experiment")
    ap.add_argument("--timeout", type=int, default=60, help="seconds per job")
    ap.add_argument("--only", nargs="+", default=["sat", "ablation", "baseline"],
                    choices=["sat", "ablation", "baseline", "size"])
    ap.add_argument("--out", default="results/results.csv")
    ap.add_argument("--new-run", action="store_true",
                    help="start a new run_id even when the configuration matches an existing run")
    args = ap.parse_args()

    if args.keep_ratios is not None and args.legacy_keep_ratio is not None:
        ap.error("use either --keep-ratio or --keep-ratios, not both")
    args.keep_ratios = args.keep_ratios or ([args.legacy_keep_ratio] if args.legacy_keep_ratio is not None else [0.4])
    del args.legacy_keep_ratio
    if not args.keep_ratios or any(not 0.0 <= ratio <= 1.0 for ratio in args.keep_ratios):
        ap.error("clue densities must be between 0 and 1")
    if len(set(args.keep_ratios)) != len(args.keep_ratios):
        ap.error("clue densities must not contain duplicates")
    if args.seeds is not None and (len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds)):
        ap.error("explicit seeds must be unique non-negative integers")
    seed_count = len(args.seeds) if args.seeds else args.puzzles
    if seed_count < 1 or not 0 <= args.tune_count < seed_count:
        ap.error("need at least one seed and 0 <= --tune-count < number of seeds")

    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)

    usable, unusable = probe_sat_solvers(SAT_SOLVER_CANDIDATES)
    for name, reason in unusable.items():
        print(f"SAT solver {name} skipped: {reason}", flush=True)
    ablation_solver = ABLATION_SOLVER if ABLATION_SOLVER in usable else (usable[0] if usable else None)
    if ablation_solver != ABLATION_SOLVER:
        print(f"ablation solver {ABLATION_SOLVER} unusable; using {ablation_solver}", flush=True)
    cfg = canonical_config(args, usable, unusable, ablation_solver)
    cfg_hash = config_hash(cfg)
    run_id = str(uuid.uuid4()) if args.new_run else cfg_hash
    csv_path, legacy_sidecar = resolve_csv_path(args.out)
    if legacy_sidecar:
        print(f"Older CSV preserved at {args.out}; writing schema v4 to {csv_path}", flush=True)
    instances = save_instances(args, out_dir)
    write_meta(os.path.join(out_dir, "meta.json"), args, usable, unusable, ablation_solver,
               instances, run_id, cfg_hash)

    jobs = build_jobs(args, usable, ablation_solver)
    for job in jobs:
        job["solver_timeout_s"] = args.timeout if job["family"] == "ilp_cp" else None
        job["worker_timeout"] = args.timeout + (30 if job["family"] == "ilp_cp" else 5)
        if job["family"] == "size":
            job["worker_timeout"] = 600
        job["process_timeout_s"] = job["worker_timeout"]
    records = {(r["box"], r["puzzle"], str(r["keep_ratio"])): r for r in instances}
    for job in jobs:
        rec = records[(job["box"], job["puzzle"], str(job["keep_ratio"]))]
        job["seed"] = rec["seed"]
        job["puzzle_sha256"] = rec["sha256"]
        job["uniqueness_checked"] = rec["uniqueness_checked"]
        job["uniqueness_status"] = rec["uniqueness_status"]
        job["solution_guarantee"] = rec["solution_guarantee"]
    done = load_done(csv_path, run_id, cfg_hash)
    jobs = [j for j in jobs if job_key(j) not in done]
    print(f"run_id={run_id} config_hash={cfg_hash}", flush=True)
    print(f"{len(jobs)} jobs to run ({len(done)} already completed in {csv_path})", flush=True)

    new_file = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for i, job in enumerate(jobs, 1):
            # Baselines enforce their own limit; the process-level timeout is only a backstop.
            timeout = job["worker_timeout"]
            res = run_job(job, timeout)
            t_enc, t_sol = res.get("t_encode", ""), res.get("t_solve", "")
            t_total = (float(t_enc) + float(t_sol)
                       if isinstance(t_enc, (int, float)) and isinstance(t_sol, (int, float)) else "")
            row = dict(schema_version=SCHEMA_VERSION, run_id=run_id, config_hash=cfg_hash,
                       git_commit=cfg["git_commit"], timeout_s=args.timeout, worker_timeout_s=timeout,
                       solver_timeout_s=(job["solver_timeout_s"] if job["solver_timeout_s"] is not None else ""),
                       process_timeout_s=job["process_timeout_s"], timeout_kind=res.get("timeout_kind", "none"),
                       t_wall=round(res.get("t_wall", 0.0), 6),
                       seed=job["seed"], puzzle_sha256=job["puzzle_sha256"],
                       split=job["split"], uniqueness_checked=job["uniqueness_checked"],
                       uniqueness_status=job["uniqueness_status"], solution_guarantee=job["solution_guarantee"],
                       experiment=job["experiment"], family=job["family"], method=job["method"],
                       solver=job["solver"], unit_alo=job["unit_alo"], box=job["box"], n=job["box"] ** 2,
                       puzzle=job["puzzle"], keep_ratio=job["keep_ratio"],
                       n_vars=res.get("n_vars", ""), n_clauses=res.get("n_clauses", ""),
                       t_encode=(round(t_enc, 6) if isinstance(t_enc, (int, float)) else ""),
                       t_solve=(round(t_sol, 6) if isinstance(t_sol, (int, float)) else ""),
                       t_total=(round(t_total, 6) if isinstance(t_total, (int, float)) else ""),
                       status=res.get("status", ""), valid=res.get("valid", ""), message=res.get("message", ""))
            writer.writerow(row)
            f.flush()
            total_text = f"{row['t_total']:.3f}s" if isinstance(row["t_total"], (int, float)) else "n/a"
            print(f"[{i}/{len(jobs)}] {job['experiment']:8s} n={row['n']:<3} {job['method']:17s} "
                  f"{job['solver']:11s} {row['status']:14s} total={total_text} wall={row['t_wall']:.3f}s "
                  f"timeout={row['timeout_kind']}", flush=True)


if __name__ == "__main__":
    main()
