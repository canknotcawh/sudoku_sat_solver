"""Bounded, explicitly configured SAT pilot for larger Sudoku grids.

This is intentionally separate from the full benchmark matrix. Every selected
encoding/solver pair is one job and the job count is capped before execution.
"""
import argparse
import csv
import hashlib
import multiprocessing as mp
import os
import signal
import time
from datetime import datetime, timezone

from src.encodings.amo import AMO_ENCODERS
from src.puzzle_io import generate_solved_grid, is_valid_solution, puncture
from src.solver import solve
from src.sudoku_encoder import SudokuEncoder

PILOT_FIELDS = ["schema_version", "started_utc", "box", "n", "puzzle", "seed", "keep_ratio",
                "puzzle_sha256", "encoding", "solver", "status", "valid", "n_vars", "n_clauses",
                "t_encode", "t_solve", "t_total", "t_wall", "peak_rss_mb", "rss_available",
                "memory_limit_mb", "timeout_s", "message"]
PILOT_SCHEMA_VERSION = 1
PILOT_STATUSES = {"solved", "timeout", "memory_limit", "encoding_only", "error"}


def puzzle_digest(givens):
    payload = ";".join(f"{r},{c},{v}" for r, c, v in sorted(givens)).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def build_pilot_jobs(box, puzzles, seeds, encodings, solvers, encode_only, keep_ratio=0.4):
    """Return exactly puzzle × encoding × solver jobs; encode-only has one sentinel solver."""
    if box < 1 or puzzles < 1 or len(seeds) != puzzles:
        raise ValueError("box/puzzles must be positive and exactly one seed is required per puzzle")
    if not encodings or set(encodings) - set(AMO_ENCODERS):
        raise ValueError(f"encodings must be selected from {sorted(AMO_ENCODERS)}")
    if encode_only:
        solver_names = ["-"]
    elif not solvers:
        raise ValueError("select at least one solver unless --encode-only is used")
    else:
        solver_names = solvers
    jobs = []
    for puzzle, seed in enumerate(seeds):
        givens = puncture(generate_solved_grid(box, seed=seed), keep_ratio=keep_ratio, seed=seed)
        for encoding in encodings:
            for solver_name in solver_names:
                jobs.append(dict(box=box, puzzle=puzzle, seed=seed, givens=givens,
                                 encoding=encoding, solver=solver_name,
                                 encode_only=encode_only, keep_ratio=keep_ratio))
    return jobs


def current_vms_bytes():
    """Current virtual size, needed to make RLIMIT_AS an additional allowance."""
    try:
        with open("/proc/self/status", encoding="ascii") as stream:
            for line in stream:
                if line.startswith("VmSize:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    return None


def peak_rss_mb(pid=None):
    """Read Linux VmHWM (peak RSS); return None where the OS interface is absent."""
    if os.name != "posix" or not os.path.exists("/proc"):
        return None
    pid = os.getpid() if pid is None else pid
    try:
        with open(f"/proc/{pid}/status", encoding="ascii") as stream:
            for line in stream:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / 1024.0
    except (OSError, ValueError):
        return None
    return None


def own_peak_rss_mb():
    """Read this process' peak RSS with OS-specific units where available."""
    value = peak_rss_mb()
    if value is not None:
        return value
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux/BSD report KiB; macOS reports bytes.
        return value / (1024 * 1024) if __import__("sys").platform == "darwin" else value / 1024.0
    except (ImportError, AttributeError, OSError, ValueError):
        return None


def _worker(conn, job, memory_limit_mb):
    result = None
    try:
        if memory_limit_mb is not None:
            try:
                import resource
                base_vms = current_vms_bytes()
                if base_vms is None or not hasattr(resource, "RLIMIT_AS"):
                    raise RuntimeError("address-space memory limit unavailable on this platform")
                limit = base_vms + int(memory_limit_mb * 1024 * 1024)
                resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
            except ImportError as exc:
                raise RuntimeError("address-space memory limit unavailable on this platform") from exc

        encoder = SudokuEncoder(box_size=job["box"], amo_method=job["encoding"], unit_alo=True)
        t0 = time.perf_counter()
        clauses, n_vars = encoder.encode(job["givens"])
        t_encode = time.perf_counter() - t0
        result = dict(status="encoding_only" if job["encode_only"] else "error", valid="",
                      n_vars=n_vars, n_clauses=len(clauses), t_encode=t_encode,
                      t_solve="", t_total=t_encode, message="")
        if not job["encode_only"]:
            t0 = time.perf_counter()
            grid = solve(encoder, clauses, solver_name=job["solver"])
            t_solve = time.perf_counter() - t0
            valid = bool(grid is not None and is_valid_solution(grid, job["givens"], job["box"]))
            result.update(status="solved" if valid else "error", valid=valid,
                          t_solve=t_solve, t_total=t_encode + t_solve,
                          message="" if valid else "solver returned no valid solution")
    except MemoryError:
        result = dict(status="memory_limit", valid="", message="worker raised MemoryError")
    except Exception as exc:
        result = dict(status="error", valid="", message=f"{type(exc).__name__}: {exc}"[:300])
    finally:
        result = result or dict(status="error", valid="", message="worker produced no result")
        result["peak_rss_mb"] = own_peak_rss_mb()
        result["rss_available"] = result["peak_rss_mb"] is not None
        conn.send(result)
        conn.close()


def run_pilot_job(job, timeout_s, memory_limit_mb=None):
    """Execute one job in a killable process, sampling peak worker RSS when available."""
    ctx = mp.get_context("fork")
    parent, child = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_worker, args=(child, job, memory_limit_mb))
    start = time.perf_counter()
    proc.start()
    child.close()
    deadline = start + timeout_s
    sampled_peak = None
    result = None
    while time.perf_counter() < deadline:
        sample = peak_rss_mb(proc.pid) if proc.is_alive() else None
        if sample is not None:
            sampled_peak = max(sampled_peak or 0.0, sample)
        if parent.poll(0.05):
            try:
                result = parent.recv()
            except EOFError:
                result = None
            break
        if not proc.is_alive():
            break
    elapsed = time.perf_counter() - start
    if result is None:
        if proc.is_alive():
            proc.terminate()
            proc.join(2)
            if proc.is_alive():
                proc.kill()
                proc.join()
            result = dict(status="timeout", valid="", message=f"process exceeded {timeout_s}s")
        elif memory_limit_mb is not None and proc.exitcode in (-signal.SIGKILL, -signal.SIGSEGV):
            result = dict(status="memory_limit", valid="", message=f"worker exited {proc.exitcode} under memory cap")
        else:
            result = dict(status="error", valid="", message=f"worker exited without result (exit={proc.exitcode})")
    proc.join(1)
    result["t_wall"] = elapsed
    post_peak = peak_rss_mb(proc.pid) if proc.pid else None
    if post_peak is not None:
        sampled_peak = max(sampled_peak or 0.0, post_peak)
    worker_peak = result.get("peak_rss_mb")
    peaks = [value for value in (sampled_peak, worker_peak) if isinstance(value, (int, float))]
    result["peak_rss_mb"] = max(peaks) if peaks else ""
    result["rss_available"] = bool(peaks)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--box", type=int, required=True, help="box side: 7=49x49, 8=64x64")
    ap.add_argument("--puzzles", type=int, default=1)
    ap.add_argument("--seeds", type=int, nargs="+", required=True, help="one seed per puzzle")
    ap.add_argument("--keep-ratio", type=float, default=0.4)
    ap.add_argument("--encoding", nargs="+", choices=sorted(AMO_ENCODERS), required=True)
    ap.add_argument("--solver", nargs="+", default=["cadical195"])
    ap.add_argument("--timeout", type=float, default=120)
    ap.add_argument("--memory-limit-mb", type=int, help="additional virtual-address-space allowance per worker")
    ap.add_argument("--encode-only", action="store_true")
    ap.add_argument("--max-jobs", type=int, default=8)
    ap.add_argument("--out", default="results/pilot.csv")
    args = ap.parse_args(argv)
    if args.timeout <= 0 or (args.memory_limit_mb is not None and args.memory_limit_mb <= 0):
        ap.error("timeout and memory limit must be positive")
    if not 0 <= args.keep_ratio <= 1:
        ap.error("keep ratio must be between 0 and 1")
    jobs = build_pilot_jobs(args.box, args.puzzles, args.seeds, args.encoding, args.solver,
                            args.encode_only, args.keep_ratio)
    if len(jobs) > args.max_jobs:
        ap.error(f"selected configuration expands to {len(jobs)} jobs, above --max-jobs={args.max_jobs}")
    output_exists = os.path.exists(args.out)
    if output_exists:
        with open(args.out, newline="", encoding="utf-8") as existing:
            existing_fields = next(csv.reader(existing), [])
        if existing_fields != PILOT_FIELDS:
            ap.error(f"existing output has an unknown pilot schema; preserving it: {args.out}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=PILOT_FIELDS)
        if not output_exists:
            writer.writeheader()
        for job in jobs:
            result = run_pilot_job(job, args.timeout, args.memory_limit_mb)
            row = dict(schema_version=PILOT_SCHEMA_VERSION, started_utc=datetime.now(timezone.utc).isoformat(),
                       box=args.box, n=args.box ** 2, puzzle=job["puzzle"], seed=job["seed"],
                       keep_ratio=job["keep_ratio"], puzzle_sha256=puzzle_digest(job["givens"]),
                       encoding=job["encoding"], solver=job["solver"], timeout_s=args.timeout,
                       memory_limit_mb=args.memory_limit_mb or "", **result)
            writer.writerow({field: row.get(field, "") for field in PILOT_FIELDS})
            stream.flush()
            print(f"{row['status']}: {args.box**2}x{args.box**2} {job['encoding']}/{job['solver']} "
                  f"wall={row['t_wall']:.2f}s rss={row['peak_rss_mb'] or 'unavailable'} MB")


if __name__ == "__main__":
    main()
