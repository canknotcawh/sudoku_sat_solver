"""Turn results/results.csv into the figures and tables used in the report, or audit the data.

    python -m src.plot_results --csv results/results.csv --out results          # figures + tables
    python -m src.plot_results --csv results/results.csv --out results --audit  # integrity checks only

Plotting writes <out>/figures/*.pdf (+ .png) and <out>/tables/*.csv and *.tex.
Timeouts are treated as right-censored at their recorded wall time; plots mark lower
bounds and display solved/timeout counts. License-capped baselines are listed in tables.
The script contains no hard-coded
results: every title, note and number is derived from the CSV.

--audit never plots. It accepts legacy v1-v3 and current v4 CSVs. It checks
duplicate job keys, status values, timing consistency, solved-row validity, and
regenerated puzzle metadata. For schema v2+ it checks run/configuration provenance.
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
from src.benchmark_schema import FIELDS, LEGACY_FIELDS, V2_FIELDS, V3_FIELDS, SCHEMA_VERSION

LIMIT = 60.0  # retained as the default value for the legacy --limit option

KEY_V1 = ["experiment", "method", "solver", "unit_alo", "box", "puzzle", "keep_ratio"]
KEY_V2 = ["run_id", "config_hash", *KEY_V1]
KEY_V3 = ["run_id", "config_hash", "seed", "puzzle_sha256", "split", *KEY_V1]
KEY_V4 = ["run_id", "config_hash", "seed", "puzzle_sha256", "timeout_kind", *KEY_V1]
KNOWN_STATUS = {"solved", "unsat", "infeasible", "timeout", "encoded", "license_limit", "missing_library",
                "memory_error", "error", "crashed"}
KNOWN_TIMEOUT_KIND = {"none", "solver", "process", "worker_crash", "worker_error"}
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
    for col in ("t_encode", "t_solve", "t_total", "t_wall", "n_vars", "n", "keep_ratio",
                "solver_timeout_s", "process_timeout_s"):
        if col not in df:
            df[col] = pd.NA
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # Older schemas have no process wall measurement; t_total is the best recorded proxy.
    df["time_observed"] = df["t_wall"].where(df["t_wall"].notna(), df["t_total"])
    df["censored"] = df["status"].eq("timeout")
    if "split" not in df:
        df["split"] = "legacy_unpartitioned"
    if "uniqueness_status" not in df:
        df["uniqueness_status"] = "legacy_unknown"
    if "timeout_kind" not in df:
        df["timeout_kind"] = df["status"].map(lambda value: "legacy_timeout" if value == "timeout" else "none")
    return df


def style_for(order, key):
    i = order.index(key)
    return SLOTS[i % len(SLOTS)], MARKERS[i % len(MARKERS)]


def fmt_time(x):
    if pd.isna(x):
        return "--"
    if x < 0.1:
        return f"{x:.3f}"
    return f"{x:.2f}" if x < 10 else f"{x:.1f}"


def fmt_aggregate(row):
    if pd.notna(row["median"]):
        value = fmt_time(row["median"])
    elif pd.notna(row["median_bound"]):
        value = ">" + fmt_time(row["median_bound"])
    else:
        value = "--"
    return f"{value} (S{int(row['n_solved'])}, T{int(row['n_timeouts'])})"


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


def kaplan_meier_median(times, events):
    """Return the KM median and its last-observation lower bound if not identified."""
    observations = sorted((float(t), bool(e)) for t, e in zip(times, events) if pd.notna(t) and float(t) >= 0)
    if not observations:
        return float("nan"), float("nan")
    survival = 1.0
    at_risk = len(observations)
    median = float("nan")
    for t in sorted({t for t, _ in observations}):
        event_count = sum(1 for time, event in observations if time == t and event)
        censored_count = sum(1 for time, event in observations if time == t and not event)
        if at_risk and event_count:
            survival *= 1.0 - event_count / at_risk
            if survival <= 0.5:
                median = t
                break
        at_risk -= event_count + censored_count
    return median, (float("nan") if pd.notna(median) else max(t for t, _ in observations))


def agg(d, keys, col="time_observed"):
    rows = []
    grouped = d.groupby(keys, dropna=False)
    for key, group in grouped:
        key = key if isinstance(key, tuple) else (key,)
        event = group.status.eq("solved")
        if "valid" in group:
            event &= group["valid"].astype(str).str.lower().eq("true")
        median, bound = kaplan_meier_median(group[col], event)
        valid = group.get("valid", pd.Series(False, index=group.index)).astype(str).str.lower().eq("true")
        solved_times = group.loc[event, col].dropna()
        row = dict(zip(keys, key))
        row.update(median=median, median_bound=bound,
                   median_plot=(median if pd.notna(median) else bound),
                   median_solved=solved_times.median() if not solved_times.empty else float("nan"),
                   q25_solved=solved_times.quantile(.25) if not solved_times.empty else float("nan"),
                   q75_solved=solved_times.quantile(.75) if not solved_times.empty else float("nan"),
                   lo=group[col].min(), hi=group[col].max(),
                   n_timeouts=int(group.status.eq("timeout").sum()),
                   n_solved=int(group.status.eq("solved").sum()), n_valid=int(valid.sum()), runs=len(group))
        rows.append(row)
    return pd.DataFrame(rows)


def evaluation_rows(df):
    if "split" not in df:
        return df
    if (df.split == "evaluation").any():
        return df[df.split == "evaluation"]
    # Historical rows have no holdout labels. Keep them usable for descriptive plots,
    # while best_sat deliberately refuses to select a winner from them.
    return df[df.split == "legacy_unpartitioned"]


def timed_sat(df):
    return evaluation_rows(df[(df.experiment == "sat") & df.status.isin(TIMED)])


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
        ax.plot(s["n"], s["median_plot"], marker=marker, color=color, linestyle=ls, label=AMO_LABEL[m],
                markeredgecolor=SURFACE, markeredgewidth=1.2)
        cens = s[s["n_timeouts"] > 0]
        ax.scatter(cens["n"], cens["median_plot"], s=70, facecolors=SURFACE, edgecolors=color, linewidths=1.6, zorder=5)
        lower = s[s["median"].isna()]
        ax.scatter(lower["n"], lower["median_bound"], marker=">", s=48, facecolors="none",
                   edgecolors=color, linewidths=1.2, zorder=6)
        for _, point in s.iterrows():
            # Labels on every AMO curve collide for the many small, similar
            # medians. Keep explicit S/T labels where censoring needs context;
            # complete counts remain available in the generated tables.
            if point.n_timeouts > 0 or pd.isna(point["median"]):
                ax.annotate(f"S{int(point.n_solved)}/T{int(point.n_timeouts)}",
                            (point.n, point.median_plot), xytext=(3, 4), textcoords="offset points", fontsize=6)
    ax.set_yscale("log")
    logx(ax, sizes_of(d))
    ax.set_xlabel("Grid side n (n x n Sudoku)")
    ax.set_ylabel("KM median worker wall time to solution (s)")
    ax.set_title(f"AMO encodings, {SOLVER_LABEL.get(solver, solver)} (timeout cases labeled; arrows are lower bounds)", loc="left")
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
            ax.bar(x, r["median_plot"], width=width - 0.03, color=color, edgecolor=SURFACE, linewidth=1.0, hatch=hatch,
                   label=SOLVER_LABEL[solver] if first else None)
            prefix = ">" if pd.isna(r["median"]) else ""
            ax.annotate(f"{prefix}S{int(r.n_solved)}/T{int(r.n_timeouts)}", (x, r.median_plot),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=5, rotation=90)
            first = False
    ax.set_yscale("log")
    ax.set_xticks(range(len(AMO_ORDER)))
    ax.set_xticklabels([AMO_LABEL[m] for m in AMO_ORDER], rotation=20, ha="right")
    ax.set_ylabel("KM median worker wall time to solution (s)")
    ax.set_title(f"Every AMO encoding x SAT solver on {n}x{n} grids (S/T labels; > marks KM lower bounds)",
                 loc="left")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=4, loc="upper left")
    fig.tight_layout()
    save(fig, out, "fig_amo_by_solver")


def best_sat(df):
    """Select per-size SAT configurations on tune seeds and report them only on evaluation seeds."""
    empty = df.iloc[0:0].copy()
    if "split" not in df or not (df.split == "tune").any() or not (df.split == "evaluation").any():
        return empty, empty
    tune = df[(df.experiment == "sat") & (df.split == "tune") & df.status.isin(TIMED)]
    evaluation = df[(df.experiment == "sat") & (df.split == "evaluation") & df.status.isin(TIMED)]
    if tune.empty or evaluation.empty:
        return empty, empty
    tuned = agg(tune, ["method", "solver", "n"])
    tuned = tuned[tuned["median"].notna()]
    if tuned.empty:
        return empty, empty
    winners = tuned.loc[tuned.groupby("n")["median"].idxmin(), ["method", "solver", "n", "median"]]
    winners = winners.rename(columns={"median": "tune_median"})
    evaluated = agg(evaluation, ["method", "solver", "n"])
    selected = winners.merge(evaluated, on=["method", "solver", "n"], how="inner")
    return selected.sort_values("n"), evaluated


def fig_vs_baselines(df, out):
    d = df[df.experiment == "baseline"]
    d = evaluation_rows(d)
    if d.empty:
        return
    best, _ = best_sat(df)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    if not best.empty:
        ax.plot(best["n"], best["median_plot"], color=INK, marker="o", linewidth=2.4,
                label="SAT (selected on tune split)", markeredgecolor=SURFACE, markeredgewidth=1.2)
        for _, point in best.iterrows():
            if point.n_timeouts > 0 or pd.isna(point["median"]):
                ax.annotate(f"S{int(point.n_solved)}/T{int(point.n_timeouts)}",
                            (point.n, point.median_plot), xytext=(3, 4), textcoords="offset points", fontsize=6)
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
        ax.plot(a["n"], a["median_plot"], marker=marker, color=color, label=label,
                markeredgecolor=SURFACE, markeredgewidth=1.2)
        cens = a[a["n_timeouts"] > 0]
        ax.scatter(cens["n"], cens["median_plot"], s=70, facecolors=SURFACE, edgecolors=color, linewidths=1.6, zorder=5)
        lower = a[a["median"].isna()]
        ax.scatter(lower["n"], lower["median_bound"], marker=">", s=48, facecolors="none",
                   edgecolors=color, linewidths=1.2, zorder=6)
        for _, point in a.iterrows():
            if point.n_timeouts > 0 or pd.isna(point["median"]):
                ax.annotate(f"S{int(point.n_solved)}/T{int(point.n_timeouts)}",
                            (point.n, point.median_plot), xytext=(3, 4), textcoords="offset points", fontsize=6)
    ax.set_yscale("log")
    logx(ax, sizes_of(d, best) if not best.empty else sizes_of(d))
    ax.set_xlabel("Grid side n (n x n Sudoku)")
    ax.set_ylabel("KM median worker wall time to solution (s)")
    title = "SAT versus ILP and CP solvers (evaluation split)"
    if best.empty:
        title += "; no tune split, SAT selection omitted"
    if capped:
        title += " (lines end where a free edition refuses the model)"
    ax.set_title(title, loc="left")
    ax.legend(loc="upper left", ncol=2, handlelength=2.2)
    fig.tight_layout()
    save(fig, out, "fig_vs_baselines")


def fig_ablation(df, out):
    off = df[(df.experiment == "ablation") & df.status.isin(TIMED)]
    off = evaluation_rows(off)
    if off.empty:
        return
    solver = off["solver"].iloc[0]  # the benchmark runs the ablation with a single SAT solver
    on = timed_sat(df)
    on = on[(on.solver == solver) & on.method.isin(ABLATION_AMOS)]
    if on.empty:
        return
    on_summary = agg(on, ["method", "n"]).set_index(["method", "n"])
    off_summary = agg(off, ["method", "n"]).set_index(["method", "n"])
    a_on, a_off = on_summary["median"], off_summary["median"]
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    for m in ABLATION_AMOS:
        keys = sorted(set(a_on.index) & set(a_off.index))
        ns = [n for (mm, n) in keys if mm == m and pd.notna(a_on[(m, n)]) and pd.notna(a_off[(m, n)])]
        if not ns:
            continue
        color, marker = style_for(AMO_ORDER, m)
        ax.plot(ns, [a_off[(m, n)] / a_on[(m, n)] for n in ns], marker=marker, color=color, label=AMO_LABEL[m],
                markeredgecolor=SURFACE, markeredgewidth=1.2)
        for n in ns:
            on_row, off_row = on_summary.loc[(m, n)], off_summary.loc[(m, n)]
            ratio = off_row["median"] / on_row["median"]
            if on_row.n_timeouts > 0 or off_row.n_timeouts > 0:
                ax.annotate(f"S{int(on_row.n_solved)}/T{int(on_row.n_timeouts)}|"
                            f"S{int(off_row.n_solved)}/T{int(off_row.n_timeouts)}",
                            (n, ratio), xytext=(3, 4), textcoords="offset points", fontsize=6)
    ax.axhline(1.0, color=INK2, linewidth=0.8, linestyle=":")
    logx(ax, sizes_of(off))
    ax.set_xlabel("Grid side n (n x n Sudoku)")
    ax.set_ylabel("KM median ratio without / with unit ALO")
    ax.set_title(f"Dropping redundant at-least-once clauses ({SOLVER_LABEL.get(solver, solver)}; S/T labels where timeouts occur)", loc="left")
    ax.legend(loc="best")
    fig.tight_layout()
    save(fig, out, "fig_ablation")


# ---------------------------------------------------------------- tables

def distribution_summary(df):
    """One row per experiment/configuration/density/split with status and time distributions."""
    group_cols = [c for c in ("experiment", "method", "solver", "n", "keep_ratio", "split") if c in df]
    distribution_rows = []
    for keys, rows in df.groupby(group_cols, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_cols, keys))
        timed = rows[rows.status.isin(TIMED)]
        valid_count = rows.valid.astype(str).str.lower().eq("true").sum()
        timeout_count = int((rows.status == "timeout").sum())
        valid_event = timed.status.eq("solved") & timed.valid.astype(str).str.lower().eq("true")
        km_median, km_bound = kaplan_meier_median(timed.time_observed, valid_event)
        solved_times = rows.loc[rows.status.eq("solved") & rows.valid.astype(str).str.lower().eq("true"),
                                "time_observed"].dropna()
        row.update(total_jobs=len(rows), valid_solutions=int(valid_count),
                   solved_jobs=int((rows.status == "solved").sum()), timeouts=timeout_count,
                   solver_timeouts=int((rows.timeout_kind == "solver").sum()),
                   process_timeouts=int((rows.timeout_kind == "process").sum()),
                   worker_crashes=int((rows.timeout_kind == "worker_crash").sum()),
                   worker_errors=int((rows.timeout_kind == "worker_error").sum()),
                   timeout_rate_all=timeout_count / len(rows) if len(rows) else 0.0,
                   km_median_to_solution_s=km_median,
                   km_median_lower_bound_s=km_bound,
                   median_solved_wall_s=solved_times.median() if not solved_times.empty else "",
                   q25_solved_wall_s=solved_times.quantile(.25) if not solved_times.empty else "",
                   q75_solved_wall_s=solved_times.quantile(.75) if not solved_times.empty else "")
        distribution_rows.append(row)
    return pd.DataFrame(distribution_rows)


def tables(df, out, summary_df=None):
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
                        row[SOLVER_LABEL[solver]] = fmt_aggregate(r)
                rows.append(row)
        write_table(pd.DataFrame(rows), out, "sat_times_table")

    b = df[df.experiment == "baseline"]
    b = evaluation_rows(b)
    if not b.empty:
        best, _ = best_sat(df)
        rows = []
        for n in sorted(b["n"].unique()):
            row = {"n": int(n)}
            br = best[best.n == n] if not best.empty else best
            row["SAT selected on tune"] = "--" if br.empty else fmt_aggregate(br.iloc[0])
            for m in BASE_ORDER:
                r = b[(b.n == n) & (b.method == m)]
                if r.empty:
                    row[BASE_LABEL[m]] = "--"
                    continue
                total, solved = len(r), int((r.status == "solved").sum())
                timeouts = int((r.status == "timeout").sum())
                ok = r[r.status.isin(TIMED)]
                if (r.status == "license_limit").all():
                    cell = f"licence cap (S{solved}, T{timeouts})"
                elif ok.empty:
                    cell = f"-- (S{solved}, T{timeouts})"
                else:
                    cell = fmt_aggregate(agg(ok, ["n"]).iloc[0])
                    if len(ok) < total:
                        cell += f"; {len(ok)}/{total} timed outcomes"
                row[BASE_LABEL[m]] = cell
            rows.append(row)
        write_table(pd.DataFrame(rows), out, "baseline_table")

    report_df = df if summary_df is None else summary_df
    group_cols = [c for c in ("experiment", "method", "solver", "n", "keep_ratio", "split") if c in report_df]
    counts = (report_df.groupby([*group_cols, "status"]).size().rename("runs").reset_index())
    write_table(counts, out, "status_counts")

    # Full distributions and denominators; timeout/license/error states remain distinguishable.
    write_table(distribution_summary(report_df), out, "result_distributions")


# ---------------------------------------------------------------- audit

def puzzle_hash(givens):
    """Same canonical hash as src.benchmark.puzzle_hash (SHA-256, first 12 hex digits)."""
    text = ";".join(f"{r},{c},{v}" for r, c, v in sorted(givens))
    return hashlib.sha256(text.encode("ascii")).hexdigest()[:12]


def audit_csv(path):
    """Return (list of failure messages, list of info lines) for the results CSV."""
    fails, info = [], []
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    columns = list(raw.columns)
    schemas = {tuple(LEGACY_FIELDS): 1, tuple(V2_FIELDS): 2, tuple(V3_FIELDS): 3, tuple(FIELDS): 4}
    if tuple(columns) not in schemas:
        fails.append(f"schema: unrecognized columns {columns}; supported v1/v2/v3/v4 schemas only")
        return fails, info
    schema = schemas[tuple(columns)]
    info.append(f"schema v{schema} ok: {len(raw)} rows, {len(columns)} columns")

    key = KEY_V1 if schema == 1 else (KEY_V2 if schema == 2 else (KEY_V3 if schema == 3 else KEY_V4))
    dup = raw[raw.duplicated(key, keep=False)]
    if not dup.empty:
        fails.append(f"{len(dup)} rows share a job key; first: {dup.iloc[0][key].to_dict()}")

    if schema >= 2:
        if (raw["schema_version"] != str(schema)).any():
            fails.append("schema_version column contains values inconsistent with the CSV header")
        required = ["run_id", "config_hash", "seed", "puzzle_sha256"]
        if schema >= 3:
            required += ["split", "uniqueness_checked", "uniqueness_status", "solution_guarantee"]
        if schema >= 4:
            required += ["timeout_kind", "t_wall", "process_timeout_s"]
        for col in required:
            if (raw[col] == "").any():
                fails.append(f"column {col}: v{schema} rows must not be empty")
        if raw["puzzle_sha256"].str.fullmatch(r"[0-9a-f]{64}").eq(False).any():
            fails.append("puzzle_sha256 contains values that are not full lowercase SHA-256 hashes")
        if raw["config_hash"].str.fullmatch(r"[0-9a-f]{64}").eq(False).any():
            fails.append("config_hash contains values that are not full lowercase SHA-256 hashes")

    bad_status = raw[~raw["status"].isin(KNOWN_STATUS)]
    if not bad_status.empty:
        fails.append(f"unknown status values: {sorted(set(bad_status['status']))}")

    numeric_cols = ["n_vars", "n_clauses", "t_encode", "t_solve", "t_total"]
    if schema >= 2:
        numeric_cols += ["schema_version", "timeout_s", "worker_timeout_s", "seed"]
    if schema >= 4:
        numeric_cols += ["t_wall", "process_timeout_s", "solver_timeout_s"]
    for col in numeric_cols:
        vals = raw[col][raw[col] != ""]
        bad = pd.to_numeric(vals, errors="coerce").isna()
        if bad.any():
            fails.append(f"column {col}: {int(bad.sum())} non-numeric values")
    if schema >= 4:
        process_limit = pd.to_numeric(raw.process_timeout_s, errors="coerce")
        worker_limit = pd.to_numeric(raw.worker_timeout_s, errors="coerce")
        if (process_limit - worker_limit).abs().gt(1e-6).any():
            fails.append("process_timeout_s must match its worker_timeout_s compatibility field")
        if process_limit.isna().any() or (process_limit <= 0).any():
            fails.append("every v4 row needs a positive process_timeout_s")
        wall = pd.to_numeric(raw.t_wall, errors="coerce")
        if wall.isna().any() or (wall < 0).any():
            fails.append("every v4 row needs a non-negative observed t_wall")
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
        fails.append(f"{len(not_valid)} solved rows without valid=True; first: {not_valid.iloc[0][key].to_dict()}")
    else:
        info.append(f"all {len(solved)} solved rows have valid=True")
    if schema >= 3:
        if not raw["split"].isin(["tune", "evaluation", "descriptive"]).all():
            fails.append("split must be tune, evaluation, or descriptive for every v3 row")
        if not raw["uniqueness_checked"].isin(["True", "False", "true", "false"]).all():
            fails.append("uniqueness_checked must be boolean for every v3 row")
        unchecked = raw[raw["uniqueness_checked"].str.lower() == "false"]
        if not unchecked.empty and not unchecked["uniqueness_status"].eq("not_checked").all():
            fails.append("unchecked puzzles must use uniqueness_status=not_checked")
        checked = raw[raw["uniqueness_checked"].str.lower() == "true"]
        if not checked.empty and not checked["uniqueness_status"].isin(["unique", "multiple"]).all():
            fails.append("checked puzzles must use uniqueness_status=unique or multiple")
        if not raw["solution_guarantee"].eq("generated_completion").all():
            fails.append("schema v3+ rows must state solution_guarantee=generated_completion")
    if schema >= 4:
        if not raw["timeout_kind"].isin(KNOWN_TIMEOUT_KIND).all():
            fails.append(f"unknown timeout_kind values: {sorted(set(raw['timeout_kind']) - KNOWN_TIMEOUT_KIND)}")
        timeout_rows = raw[raw.status == "timeout"]
        if not timeout_rows["timeout_kind"].isin(["solver", "process"]).all():
            fails.append("v4 timeout rows must identify solver or process timeout")
        timeout_kinds = raw.timeout_kind.isin(["solver", "process"])
        if (timeout_kinds & raw.status.ne("timeout")).any():
            fails.append("solver/process timeout_kind may only appear on timeout rows")
        solver_timeouts = raw[raw.timeout_kind == "solver"]
        if solver_timeouts[["t_encode", "t_solve", "t_total", "solver_timeout_s"]].eq("").any().any():
            fails.append("solver-timeout rows must retain build/solver timing and the solver limit")
        solver_limits = pd.to_numeric(solver_timeouts.solver_timeout_s, errors="coerce")
        if solver_limits.isna().any() or (solver_limits <= 0).any():
            fails.append("solver-timeout rows need a positive solver_timeout_s")
        process_timeouts = raw[raw.timeout_kind == "process"]
        if process_timeouts["t_wall"].eq("").any():
            fails.append("process-timeout rows must record observed t_wall")
        process_limits = pd.to_numeric(process_timeouts.process_timeout_s, errors="coerce")
        if process_limits.isna().any() or (process_limits <= 0).any():
            fails.append("process-timeout rows need a positive process_timeout_s")
        if not raw.loc[raw.status == "crashed", "timeout_kind"].eq("worker_crash").all():
            fails.append("crashed rows must use timeout_kind=worker_crash")
        phase_cols = ["t_encode", "t_solve", "t_total"]
        phase = raw[phase_cols].apply(pd.to_numeric, errors="coerce")
        phase_complete = phase.notna().all(axis=1)
        phase_mismatch = (phase.loc[phase_complete, "t_total"] -
                          (phase.loc[phase_complete, "t_encode"] + phase.loc[phase_complete, "t_solve"])).abs() > 1e-3
        if phase_mismatch.any():
            fails.append(f"{int(phase_mismatch.sum())} v4 rows where t_total != t_encode + t_solve")
        if (raw.loc[process_timeouts.index, phase_cols] != "").any().any():
            fails.append("process-timeout rows must leave unavailable phase timings blank")
        crashed = raw[raw.status == "crashed"]
        if (crashed[phase_cols] != "").any().any():
            fails.append("worker-crash rows must leave unavailable phase timings blank")
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
    legacy_records = {}
    record_entries = {}
    run_records = {}
    run_configs = {}
    for run in meta.get("runs", []):
        if run.get("run_id") and run.get("config"):
            expected_hash = hashlib.sha256(json.dumps(
                run["config"], sort_keys=True, separators=(",", ":"), ensure_ascii=True
            ).encode("utf-8")).hexdigest()
            if run.get("config_hash") != expected_hash:
                fails.append(f"run {run['run_id']}: config_hash does not match recorded config")
            run_configs[run["run_id"]] = (run.get("config_hash"), run.get("git_commit"))
        for rec in run.get("instances", []):
            key = (rec["box"], rec["puzzle"], str(rec["keep_ratio"]))
            if run.get("run_id"):
                identity = ("run", run["run_id"], *key)
                previous = run_records.get((run["run_id"], *key))
                if previous and previous.get("sha256") != rec.get("sha256"):
                    fails.append(f"run {run['run_id']} instance {key} has conflicting puzzle hashes")
                run_records[(run["run_id"], *key)] = rec
            else:
                identity = ("legacy", *key)
                if key in legacy_records and legacy_records[key]["sha256_12"] != rec["sha256_12"]:
                    fails.append(f"legacy instance {key} recorded with two different hashes across runs")
                legacy_records[key] = rec
            record_entries[identity] = (key, rec)

    checked = 0
    for _, (key, rec) in sorted(record_entries.items()):
        box = rec["box"]
        regenerated = puncture(generate_solved_grid(box, seed=rec["seed"]), keep_ratio=rec["keep_ratio"],
                               seed=rec["seed"])
        regenerated_hash = puzzle_hash(regenerated)
        if regenerated_hash != rec["sha256_12"] or len(regenerated) != rec["n_clues"]:
            fails.append(f"instance {key}: regenerating from seed {rec['seed']} does not reproduce the recorded hash")
        full_hash = hashlib.sha256(";".join(
            f"{r},{c},{v}" for r, c, v in sorted(regenerated)
        ).encode("ascii")).hexdigest()
        if rec.get("sha256") and rec["sha256"] != full_hash:
            fails.append(f"instance {key}: full SHA-256 does not match regenerated puzzle")
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
                f"{' and saved files' if any(r.get('file') for _, r in record_entries.values()) else ''})")

    raw = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    if list(raw.columns) in (V2_FIELDS, V3_FIELDS, FIELDS):
        used = raw
        missing = []
        bad_hashes = []
        for row in used.to_dict("records"):
            key = (row["run_id"], int(row["box"]), int(row["puzzle"]), row["keep_ratio"])
            rec = run_records.get(key)
            if rec is None:
                missing.append(key)
            elif row["puzzle_sha256"] != rec.get("sha256") or int(row["seed"]) != rec["seed"]:
                bad_hashes.append(key)
            elif list(raw.columns) in (V3_FIELDS, FIELDS) and (
                (row["experiment"] != "size" and row["split"] != rec.get("split"))
                or row["uniqueness_checked"].lower() != str(rec.get("uniqueness_checked")).lower()
                or row["uniqueness_status"] != rec.get("uniqueness_status")
                or row["solution_guarantee"] != rec.get("solution_guarantee")
            ):
                bad_hashes.append(key + ("split/uniqueness",))
            run_config = run_configs.get(row["run_id"])
            if run_config is None or row["config_hash"] != run_config[0] or row["git_commit"] != (run_config[1] or ""):
                bad_hashes.append(key + ("config/commit",))
        if missing:
            fails.append(f"{len(missing)} provenance rows have no matching run-specific puzzle metadata, e.g. {missing[:3]}")
        if bad_hashes:
            fails.append(f"{len(bad_hashes)} provenance rows disagree with recorded puzzle metadata, e.g. {bad_hashes[:3]}")
    else:
        used = raw[raw.experiment.isin(["sat", "ablation", "baseline"])]
        used_keys = {(int(b), int(p), k) for b, p, k in zip(used.box, used.puzzle, used.keep_ratio)}
        missing = sorted(k for k in used_keys if k not in legacy_records)
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
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="results/results.csv")
    ap.add_argument("--out", default="results")
    ap.add_argument("--limit", type=float, help="deprecated compatibility option; observed times are no longer capped")
    ap.add_argument("--run-id", help="plot only this provenance run_id (defaults to the last run in the CSV)")
    ap.add_argument("--keep-ratio", type=float, help="select one clue density when the CSV contains several")
    ap.add_argument("--audit", action="store_true",
                    help="check the CSV and the saved instances instead of plotting; exit 1 on failure")
    args = ap.parse_args()
    if args.limit is not None:
        print("--limit is deprecated and ignored; timeout censoring uses recorded wall times and limits.")

    if args.audit:
        sys.exit(0 if run_audit(args.csv, args.out) else 1)

    style()
    df = load(args.csv)
    if "run_id" in df.columns:
        available = list(df["run_id"].dropna().astype(str).unique())
        selected = args.run_id or (available[-1] if available else None)
        if selected is not None and selected not in available:
            ap.error(f"run_id {selected!r} not found in {args.csv}; available: {available}")
        if selected is not None:
            df = df[df["run_id"].astype(str) == selected]
            print(f"plotting run_id={selected}")
    all_densities = df.copy()
    ratios = sorted(df["keep_ratio"].dropna().unique())
    if args.keep_ratio is not None:
        if args.keep_ratio not in ratios:
            ap.error(f"keep_ratio {args.keep_ratio} not found; available: {ratios}")
        df = df[df.keep_ratio == args.keep_ratio]
    elif len(ratios) > 1:
        ap.error(f"multiple clue densities present; specify --keep-ratio from {ratios}")
    fig_size(df, args.out)
    fig_amo_vs_n(df, args.out)
    fig_amo_by_solver(df, args.out)
    fig_vs_baselines(df, args.out)
    fig_ablation(df, args.out)
    tables(df, args.out, summary_df=all_densities)
    print("figures and tables written to", args.out)


if __name__ == "__main__":
    main()
