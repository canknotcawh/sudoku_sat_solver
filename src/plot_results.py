"""Turn results/results.csv into the figures and tables used in the report, or audit the data.

    python -m src.plot_results --csv results/results.csv --out results          # figures + tables
    python -m src.plot_results --csv results/results.csv --out results --audit  # integrity checks only

Plotting writes <out>/figures/*.pdf (+ .png) and <out>/tables/*.csv and *.tex.
Timeouts are drawn at the time limit with an open marker; license-capped baselines are
listed in the tables rather than plotted as zeros. The script contains no hard-coded
results: every title, note and number is derived from the CSV.

--audit never plots. It checks the CSV schema, duplicate job keys, status values, timing
consistency, that every "solved" row has valid=True, and (when <out>/meta.json exists)
that the saved puzzles in <out>/instances/ parse, match their recorded content hash,
regenerate from their recorded seed, and cover every puzzle referenced by the CSV.
The exit code is 1 if any check fails.
"""
import argparse
import hashlib
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import pandas as pd  # noqa: E402

LIMIT = 60.0  # seconds, the benchmark time limit (overridable with --limit)

FIELDS = ["experiment", "family", "method", "solver", "unit_alo", "box", "n", "puzzle", "keep_ratio",
          "n_vars", "n_clauses", "t_encode", "t_solve", "t_total", "status", "valid", "message"]
KEY = ["experiment", "method", "solver", "unit_alo", "box", "puzzle", "keep_ratio"]
KNOWN_STATUS = {"solved", "unsat", "infeasible", "timeout", "encoded", "license_limit", "missing_library",
                "memory_error", "error", "crashed"}
TIMED = ("solved", "timeout")  # statuses that carry a meaningful time

# Categorical slots in the fixed order of the validated palette; a method keeps its colour everywhere.
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "<"]

AMO_ORDER = ["pairwise", "sequential", "commander", "binary", "product",
             "pysat_seqcounter", "pysat_bitwise", "pysat_ladder"]
AMO_LABEL = {"pairwise": "pairwise", "sequential": "sequential", "commander": "commander", "binary": "binary",
             "product": "product", "pysat_seqcounter": "PySAT seqcounter", "pysat_bitwise": "PySAT bitwise",
             "pysat_ladder": "PySAT ladder"}
BASE_ORDER = ["cpsat", "gurobi", "cplex_mip", "cplex_cp", "cbc_ilp"]
BASE_LABEL = {"cpsat": "CP-SAT", "gurobi": "Gurobi (ILP)", "cplex_mip": "CPLEX MIP", "cplex_cp": "CPLEX CP",
              "cbc_ilp": "CBC (ILP)"}
SOLVER_ORDER = ["cadical195", "glucose42", "kissat404", "minisat22"]
SOLVER_LABEL = {"cadical195": "CaDiCaL 1.9.5", "glucose42": "Glucose 4.2", "kissat404": "Kissat 4.0.4",
                "minisat22": "MiniSat 2.2"}
ABLATION_AMOS = ["pairwise", "sequential", "product"]


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.titlecolor": INK, "axes.titlesize": 10, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.axisbelow": True, "lines.linewidth": 1.8, "lines.markersize": 6,
        "legend.frameon": False, "pdf.fonttype": 42,
    })


def logx(ax, ticks):
    """Log x axis labelled only at the grid sizes that were run."""
    ax.set_xscale("log")
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())


def sizes_of(*frames):
    """Sorted grid sides present in the given frames (used for the x ticks)."""
    return sorted({int(n) for f in frames for n in f["n"].dropna().unique()})


def save(fig, out, name):
    os.makedirs(os.path.join(out, "figures"), exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, "figures", f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def load(path):
    df = pd.read_csv(path)
    for col in ("t_encode", "t_solve", "t_total", "n_vars", "n_clauses", "n"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["censored"] = (df["status"] == "timeout") | (df["t_total"] > LIMIT)
    df["t_plot"] = df["t_total"].where(~df["censored"], LIMIT)
    return df


def style_for(order, key):
    i = order.index(key)
    return SLOTS[i % len(SLOTS)], MARKERS[i % len(MARKERS)]


def fmt_time(x, censored=False):
    if pd.isna(x):
        return "--"
    if censored:
        return f">{LIMIT:.0f}"
    return f"{x:.2f}" if x < 10 else f"{x:.1f}"


def tex_escape(s):
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("#", r"\#")


def write_table(df, out, name):
    os.makedirs(os.path.join(out, "tables"), exist_ok=True)
    df.to_csv(os.path.join(out, "tables", f"{name}.csv"), index=False)
    cols = "l" + "r" * (len(df.columns) - 1)
    lines = [rf"\begin{{tabular}}{{{cols}}}", r"\toprule",
             " & ".join(tex_escape(c) for c in df.columns) + r" \\", r"\midrule"]
    for _, row in df.iterrows():
        lines.append(" & ".join(tex_escape(v) for v in row.tolist()) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(out, "tables", f"{name}.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")


def agg(d, keys, col="t_plot"):
    g = d.groupby(keys)[col]
    out = g.median().rename("median").to_frame()
    out["lo"], out["hi"] = g.min(), g.max()
    out["n_timeouts"] = d.groupby(keys)["censored"].sum()
    out["runs"] = d.groupby(keys)["censored"].size()
    return out.reset_index()


def timed_sat(df):
    return df[(df.experiment == "sat") & df.status.isin(TIMED)]


# ---------------------------------------------------------------- figures

def fig_size(df, out):
    d = df[df.experiment == "size"]
    if d.empty:
        return
    panels = [("n_clauses", "Clauses"), ("n_vars", "Variables"), ("t_encode", "Encoding time (s)")]
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.2))
    for ax, (col, title) in zip(axes, panels):
        for m in AMO_ORDER:
            s = d[d.method == m].sort_values("n")
            if s.empty:
                continue
            color, marker = style_for(AMO_ORDER, m)
            ls = "--" if m.startswith("pysat_") else "-"
            ax.plot(s["n"], s[col], marker=marker, color=color, linestyle=ls, label=AMO_LABEL[m],
                    markeredgecolor=SURFACE, markeredgewidth=1.2)
        ax.set_yscale("log")
        logx(ax, sizes_of(d))
        ax.set_xlabel("Grid side n (n x n Sudoku)")
        ax.set_title(title, loc="left")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=8, bbox_to_anchor=(0.5, -0.08), columnspacing=1.2,
               handlelength=1.8)
    fig.tight_layout()
    save(fig, out, "fig_encoding_size")


def fig_amo_vs_n(df, out, solver="cadical195"):
    d = timed_sat(df)
    if solver not in set(d["solver"]):
        solver = d["solver"].iloc[0] if not d.empty else None
    if solver is None:
        return
    d = d[d.solver == solver]
    a = agg(d, ["method", "n"])
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for m in AMO_ORDER:
        s = a[a.method == m].sort_values("n")
        if s.empty:
            continue
        color, marker = style_for(AMO_ORDER, m)
        ls = "--" if m.startswith("pysat_") else "-"
        ax.plot(s["n"], s["median"], marker=marker, color=color, linestyle=ls, label=AMO_LABEL[m],
                markeredgecolor=SURFACE, markeredgewidth=1.2)
        cens = s[s["n_timeouts"] > 0]
        ax.scatter(cens["n"], cens["median"], s=70, facecolors=SURFACE, edgecolors=color, linewidths=1.6, zorder=5)
    ax.axhline(LIMIT, color=INK2, linewidth=0.8, linestyle=":")
    ax.text(ax.get_xlim()[0], LIMIT * 1.08, "time limit", color=INK2, fontsize=8)
    ax.set_yscale("log")
    logx(ax, sizes_of(d))
    ax.set_xlabel("Grid side n (n x n Sudoku)")
    ax.set_ylabel("Median time, encode + solve (s)")
    ax.set_title(f"AMO encodings, {SOLVER_LABEL.get(solver, solver)} (open marker: a run hit the limit)", loc="left")
    ax.legend(ncol=2, loc="upper left", handlelength=2.2)
    fig.tight_layout()
    save(fig, out, "fig_amo_vs_n")


def fig_amo_by_solver(df, out, n=None):
    d = timed_sat(df)
    if d.empty:
        return
    n = n or int(d["n"].max())
    d = d[d["n"] == n]
    solvers = [s for s in SOLVER_ORDER if s in set(d["solver"])]
    a = agg(d, ["method", "solver"])
    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    width = 0.8 / max(len(solvers), 1)
    for j, solver in enumerate(solvers):
        color = SLOTS[SOLVER_ORDER.index(solver)]
        first = True
        for i, m in enumerate(AMO_ORDER):
            r = a[(a.method == m) & (a.solver == solver)]
            if r.empty:
                continue
            x = i + (j - (len(solvers) - 1) / 2) * width
            r = r.iloc[0]
            hatch = "///" if r["n_timeouts"] > 0 else None
            ax.bar(x, r["median"], width=width - 0.03, color=color, edgecolor=SURFACE, linewidth=1.0, hatch=hatch,
                   label=SOLVER_LABEL[solver] if first else None)
            first = False
    ax.set_yscale("log")
    ax.set_xticks(range(len(AMO_ORDER)))
    ax.set_xticklabels([AMO_LABEL[m] for m in AMO_ORDER], rotation=20, ha="right")
    ax.set_ylabel("Median time, encode + solve (s)")
    ax.set_title(f"Every AMO encoding x SAT solver on {n}x{n} grids (hatched: at least one run hit the limit)",
                 loc="left")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=4, loc="upper left")
    fig.tight_layout()
    save(fig, out, "fig_amo_by_solver")


def best_sat(df):
    """Best SAT configuration per grid size: lowest median total time over (AMO, solver)."""
    d = timed_sat(df)
    if d.empty:
        return d, d
    a = agg(d, ["method", "solver", "n"])
    idx = a.groupby("n")["median"].idxmin()
    return a.loc[idx].sort_values("n"), a


def fig_vs_baselines(df, out):
    d = df[df.experiment == "baseline"]
    if d.empty:
        return
    best, _ = best_sat(df)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    if not best.empty:
        ax.plot(best["n"], best["median"], color=INK, marker="o", linewidth=2.4,
                label="SAT (best AMO + solver per size)", markeredgecolor=SURFACE, markeredgewidth=1.2)
    capped = False
    for m in BASE_ORDER:
        s = d[(d.method == m) & d.status.isin(TIMED)]
        lic = d[(d.method == m) & (d.status == "license_limit")]
        label = BASE_LABEL[m]
        if not lic.empty:
            label += f" (cap from n={int(lic['n'].min())})"
            capped = True
        if s.empty:
            continue
        a = agg(s, ["n"]).sort_values("n")
        color, marker = style_for(BASE_ORDER, m)
        ax.plot(a["n"], a["median"], marker=marker, color=color, label=label,
                markeredgecolor=SURFACE, markeredgewidth=1.2)
        cens = a[a["n_timeouts"] > 0]
        ax.scatter(cens["n"], cens["median"], s=70, facecolors=SURFACE, edgecolors=color, linewidths=1.6, zorder=5)
    ax.axhline(LIMIT, color=INK2, linewidth=0.8, linestyle=":")
    ax.set_yscale("log")
    logx(ax, sizes_of(d, best) if not best.empty else sizes_of(d))
    ax.set_xlabel("Grid side n (n x n Sudoku)")
    ax.set_ylabel("Median time, build + solve (s)")
    title = "SAT versus ILP and CP solvers"
    if capped:
        title += " (lines end where a free edition refuses the model)"
    ax.set_title(title, loc="left")
    ax.legend(loc="upper left", ncol=2, handlelength=2.2)
    fig.tight_layout()
    save(fig, out, "fig_vs_baselines")


def fig_ablation(df, out):
    off = df[(df.experiment == "ablation") & df.status.isin(TIMED)]
    if off.empty:
        return
    solver = off["solver"].iloc[0]  # the benchmark runs the ablation with a single SAT solver
    on = timed_sat(df)
    on = on[(on.solver == solver) & on.method.isin(ABLATION_AMOS)]
    if on.empty:
        return
    a_on = agg(on, ["method", "n"]).set_index(["method", "n"])["median"]
    a_off = agg(off, ["method", "n"]).set_index(["method", "n"])["median"]
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    for m in ABLATION_AMOS:
        keys = sorted(set(a_on.index) & set(a_off.index))
        ns = [n for (mm, n) in keys if mm == m]
        if not ns:
            continue
        color, marker = style_for(AMO_ORDER, m)
        ax.plot(ns, [a_off[(m, n)] / a_on[(m, n)] for n in ns], marker=marker, color=color, label=AMO_LABEL[m],
                markeredgecolor=SURFACE, markeredgewidth=1.2)
    ax.axhline(1.0, color=INK2, linewidth=0.8, linestyle=":")
    logx(ax, sizes_of(off))
    ax.set_xlabel("Grid side n (n x n Sudoku)")
    ax.set_ylabel("Time without / with unit ALO")
    ax.set_title(f"Dropping the redundant at-least-once clauses ({SOLVER_LABEL.get(solver, solver)})", loc="left")
    ax.legend(loc="best")
    fig.tight_layout()
    save(fig, out, "fig_ablation")


# ---------------------------------------------------------------- tables

def tables(df, out):
    s = df[df.experiment == "size"]
    if not s.empty:
        rows = []
        for n in sorted(s["n"].unique()):
            row = {"n": int(n)}
            for m in AMO_ORDER:
                r = s[(s.n == n) & (s.method == m)]
                row[AMO_LABEL[m] + " cls"] = "" if r.empty else f"{int(r.iloc[0].n_clauses):,}"
                row[AMO_LABEL[m] + " vars"] = "" if r.empty else f"{int(r.iloc[0].n_vars):,}"
            rows.append(row)
        write_table(pd.DataFrame(rows), out, "size_table")

    d = timed_sat(df)
    if not d.empty:
        a = agg(d, ["method", "solver", "n"])
        rows = []
        for n in sorted(a["n"].unique()):
            for m in AMO_ORDER:
                row = {"n": int(n), "AMO": AMO_LABEL[m]}
                for solver in SOLVER_ORDER:
                    r = a[(a.n == n) & (a.method == m) & (a.solver == solver)]
                    if r.empty:
                        row[SOLVER_LABEL[solver]] = "--"
                    else:
                        r = r.iloc[0]
                        row[SOLVER_LABEL[solver]] = fmt_time(r["median"], r["n_timeouts"] == r["runs"])
                rows.append(row)
        write_table(pd.DataFrame(rows), out, "sat_times_table")

    b = df[df.experiment == "baseline"]
    if not b.empty:
        best, _ = best_sat(df)
        rows = []
        for n in sorted(b["n"].unique()):
            row = {"n": int(n)}
            br = best[best.n == n] if not best.empty else best
            row["SAT best"] = "--" if br.empty else fmt_time(br.iloc[0]["median"])
            for m in BASE_ORDER:
                r = b[(b.n == n) & (b.method == m)]
                if r.empty:
                    row[BASE_LABEL[m]] = "--"
                elif (r.status == "license_limit").all():
                    row[BASE_LABEL[m]] = "licence cap"
                elif (r.status == "timeout").all():
                    row[BASE_LABEL[m]] = f">{LIMIT:.0f}"
                else:
                    ok = r[r.status.isin(TIMED)]
                    row[BASE_LABEL[m]] = fmt_time(ok["t_plot"].median()) if not ok.empty else "--"
            rows.append(row)
        write_table(pd.DataFrame(rows), out, "baseline_table")

    counts = (df.groupby(["experiment", "method", "n", "status"]).size().rename("runs").reset_index())
    write_table(counts, out, "status_counts")


# ---------------------------------------------------------------- audit

def puzzle_hash(givens):
    """Same canonical hash as src.benchmark.puzzle_hash (SHA-256, first 12 hex digits)."""
    text = ";".join(f"{r},{c},{v}" for r, c, v in sorted(givens))
    return hashlib.sha256(text.encode("ascii")).hexdigest()[:12]


def audit_csv(path):
    """Return (list of failure messages, list of info lines) for the results CSV."""
    fails, info = [], []
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    if list(raw.columns) != FIELDS:
        fails.append(f"schema: columns {list(raw.columns)} differ from expected {FIELDS}")
        return fails, info
    info.append(f"schema ok: {len(raw)} rows, {len(FIELDS)} columns")

    dup = raw[raw.duplicated(KEY, keep=False)]
    if not dup.empty:
        fails.append(f"{len(dup)} rows share a job key; first: {dup.iloc[0][KEY].to_dict()}")

    bad_status = raw[~raw["status"].isin(KNOWN_STATUS)]
    if not bad_status.empty:
        fails.append(f"unknown status values: {sorted(set(bad_status['status']))}")

    for col in ("n_vars", "n_clauses", "t_encode", "t_solve", "t_total"):
        vals = raw[col][raw[col] != ""]
        bad = pd.to_numeric(vals, errors="coerce").isna()
        if bad.any():
            fails.append(f"column {col}: {int(bad.sum())} non-numeric values")
    num = raw.assign(**{c: pd.to_numeric(raw[c], errors="coerce") for c in ("box", "n", "t_encode", "t_solve", "t_total")})
    if (num["n"] != num["box"] ** 2).any():
        fails.append("rows where n != box^2")
    mismatch = num[(num.t_total - (num.t_encode.fillna(0) + num.t_solve.fillna(0))).abs() > 1e-3]
    if not mismatch.empty:
        fails.append(f"{len(mismatch)} rows where t_total != t_encode + t_solve")
    if (num[["t_encode", "t_solve", "t_total"]] < 0).any().any():
        fails.append("negative timings")

    solved = raw[raw.status == "solved"]
    not_valid = solved[solved["valid"] != "True"]
    if not not_valid.empty:
        fails.append(f"{len(not_valid)} solved rows without valid=True; first: {not_valid.iloc[0][KEY].to_dict()}")
    else:
        info.append(f"all {len(solved)} solved rows have valid=True")
    info.append("status counts: " + ", ".join(f"{k}={v}" for k, v in raw["status"].value_counts().items()))
    return fails, info


def audit_instances(csv_path, meta_path):
    """Check saved puzzles against meta.json and against the CSV."""
    fails, info = [], []
    if not os.path.exists(meta_path):
        return [f"{meta_path} not found: cannot verify instances"], info
    try:
        from src.puzzle_io import generate_solved_grid, parse_string, puncture
    except ImportError as e:  # pragma: no cover
        return [f"cannot import src.puzzle_io: {e}"], info

    with open(meta_path) as f:
        meta = json.load(f)
    out_dir = os.path.dirname(meta_path)
    records = {}
    for run in meta.get("runs", []):
        for rec in run.get("instances", []):
            key = (rec["box"], rec["puzzle"], str(rec["keep_ratio"]))
            if key in records and records[key]["sha256_12"] != rec["sha256_12"]:
                fails.append(f"instance {key} recorded with two different hashes across runs")
            records[key] = rec

    checked = 0
    for key, rec in sorted(records.items()):
        box = rec["box"]
        regenerated = puncture(generate_solved_grid(box, seed=rec["seed"]), keep_ratio=rec["keep_ratio"],
                               seed=rec["seed"])
        if puzzle_hash(regenerated) != rec["sha256_12"] or len(regenerated) != rec["n_clues"]:
            fails.append(f"instance {key}: regenerating from seed {rec['seed']} does not reproduce the recorded hash")
        if rec.get("file"):
            path = os.path.join(out_dir, rec["file"])
            if not os.path.exists(path):
                fails.append(f"instance {key}: file {rec['file']} missing")
                continue
            try:
                with open(path) as f:
                    givens = parse_string(f.read(), box_size=box)
            except ValueError as e:
                fails.append(f"instance {key}: {rec['file']} does not parse: {e}")
                continue
            if puzzle_hash(givens) != rec["sha256_12"] or len(givens) != rec["n_clues"]:
                fails.append(f"instance {key}: {rec['file']} does not match its recorded hash / clue count")
        checked += 1
    info.append(f"{checked} puzzles in meta.json re-verified (seed regeneration"
                f"{' and saved files' if any(r.get('file') for r in records.values()) else ''})")

    raw = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    used = raw[raw.experiment.isin(["sat", "ablation", "baseline"])]
    used_keys = {(int(b), int(p), k) for b, p, k in zip(used.box, used.puzzle, used.keep_ratio)}
    missing = sorted(k for k in used_keys if k not in records)
    if missing:
        fails.append(f"{len(missing)} puzzles used in the CSV have no record in meta.json, e.g. {missing[:3]}")
    return fails, info


def run_audit(csv_path, out_dir):
    fails, info = audit_csv(csv_path)
    if not fails or "schema" not in fails[0]:
        f2, i2 = audit_instances(csv_path, os.path.join(out_dir, "meta.json"))
        fails += f2
        info += i2
    for line in info:
        print("ok    ", line)
    for line in fails:
        print("FAIL  ", line)
    print("AUDIT PASSED" if not fails else f"AUDIT FAILED ({len(fails)} problems)")
    return not fails


# ---------------------------------------------------------------- main

def main():
    global LIMIT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="results/results.csv")
    ap.add_argument("--out", default="results")
    ap.add_argument("--limit", type=float, default=LIMIT, help="time limit (s) used for the benchmark run")
    ap.add_argument("--audit", action="store_true",
                    help="check the CSV and the saved instances instead of plotting; exit 1 on failure")
    args = ap.parse_args()
    LIMIT = args.limit

    if args.audit:
        sys.exit(0 if run_audit(args.csv, args.out) else 1)

    style()
    df = load(args.csv)
    fig_size(df, args.out)
    fig_amo_vs_n(df, args.out)
    fig_amo_by_solver(df, args.out)
    fig_vs_baselines(df, args.out)
    fig_ablation(df, args.out)
    tables(df, args.out)
    print("figures and tables written to", args.out)


if __name__ == "__main__":
    main()