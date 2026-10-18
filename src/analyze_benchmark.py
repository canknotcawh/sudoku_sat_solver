"""Audit and summarize benchmark raw CSVs without changing them.

Uses only the Python standard library. Outputs CSV summary tables, Markdown
report, and a simple SVG chart. Missing timing values remain missing.
"""
import argparse
import csv
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

REQUIRED = {
    "run_id", "instance_id", "instance_path", "seed", "box_size", "grid_size",
    "clue_count", "amo_method", "solver", "backend", "status", "solved",
    "timeout", "time_limit_s", "n_vars", "n_clauses", "t_encode", "t_solve", "version",
}
AMO_METHODS = ("pairwise", "sequential", "commander", "binary", "product")
STATUSES = ("SOLVED", "UNSAT", "TIMEOUT", "UNKNOWN", "BACKEND_UNAVAILABLE", "INVALID_SOLUTION", "ERROR")
SUMMARY_FIELDS = (
    "dataset", "grid_size", "backend", "amo_method", "solver", "n_rows", "n_instances",
    "n_solved", "n_unsat", "n_timeout", "n_unknown", "n_backend_unavailable", "n_invalid", "n_error",
    "solve_median_s", "solve_iqr_s", "encode_median_s", "encode_iqr_s", "n_vars_median",
    "n_clauses_median", "time_limit_s",
)


def number(value):
    if value is None or value == "":
        return None
    return float(value)


def quantile(values, fraction):
    """Linear interpolation at (n - 1) * fraction, matching inclusive quartiles."""
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def median_iqr(values):
    if not values:
        return "", ""
    return f"{statistics.median(values):.9f}", f"{quantile(values, .75) - quantile(values, .25):.9f}"


def read_dataset(label, path):
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        headers = set(reader.fieldnames or ())
        rows = list(reader)
    missing_columns = sorted(REQUIRED - headers)
    extra_columns = sorted(headers - REQUIRED)
    issues = []
    if missing_columns:
        issues.append(f"missing required columns: {', '.join(missing_columns)}")
    for index, row in enumerate(rows, start=2):
        if None in row:
            issues.append(f"row {index}: extra CSV fields")
        for column in REQUIRED:
            if column in headers and (row.get(column) is None or row.get(column) == ""):
                # These fields are intentionally blank for unavailable backends / not-applicable backend.
                allowed = column in {"amo_method", "n_vars", "n_clauses", "t_encode", "t_solve"}
                if not allowed:
                    issues.append(f"row {index}: required value {column} is blank")
    keys = Counter((r.get("run_id"), r.get("instance_id"), r.get("backend"), r.get("amo_method"), r.get("solver")) for r in rows)
    duplicates = [key for key, count in keys.items() if count > 1]
    if duplicates:
        issues.append(f"duplicate configuration keys: {len(duplicates)}")
    return {"label": label, "path": path, "rows": rows, "headers": headers,
            "missing_columns": missing_columns, "extra_columns": extra_columns,
            "duplicates": duplicates, "issues": issues}


def audit_dataset(data):
    rows = data["rows"]
    status_counts = Counter(r["status"] for r in rows)
    backend_counts = Counter(r["backend"] for r in rows)
    ids = defaultdict(list)
    for row in rows:
        ids[row["instance_id"]].append(row)
        if row["solved"].lower() not in {"true", "false"}:
            data["issues"].append(f"{row['instance_id']}: invalid solved value {row['solved']!r}")
        if row["timeout"].lower() not in {"true", "false"}:
            data["issues"].append(f"{row['instance_id']}: invalid timeout value {row['timeout']!r}")
        if row["status"] == "SOLVED" and row["solved"].lower() != "true":
            data["issues"].append(f"{row['instance_id']}: SOLVED but solved is false")
        if row["status"] == "UNSAT" and row["solved"].lower() != "false":
            data["issues"].append(f"{row['instance_id']}: UNSAT but solved is true")
        if row["status"] == "TIMEOUT" and row["timeout"].lower() != "true":
            data["issues"].append(f"{row['instance_id']}: TIMEOUT but timeout is false")
        if row["status"] in {"UNSAT", "TIMEOUT", "UNKNOWN", "SOLVED"} and row["t_solve"] == "":
            data["issues"].append(f"{row['instance_id']}: {row['status']} has no solve time")
        for col in ("time_limit_s", "n_vars", "n_clauses", "t_encode", "t_solve"):
            if row.get(col):
                try:
                    if number(row[col]) < 0:
                        data["issues"].append(f"{row['instance_id']}: negative {col}")
                except ValueError:
                    data["issues"].append(f"{row['instance_id']}: nonnumeric {col}={row[col]!r}")
    for instance_id, group in ids.items():
        shared = ("seed", "box_size", "grid_size", "clue_count", "instance_path", "time_limit_s")
        for field in shared:
            if len({row[field] for row in group}) != 1:
                data["issues"].append(f"{instance_id}: inconsistent {field} across configurations")
        sat_rows = [r for r in group if r["backend"] == "pysat"]
        if sat_rows:
            methods = [r["amo_method"] for r in sat_rows]
            if len(methods) != len(AMO_METHODS) or set(methods) != set(AMO_METHODS):
                data["issues"].append(f"{instance_id}: expected one row per AMO method; got {methods}")
        paths = {r["instance_path"] for r in group}
        for raw_path in paths:
            puzzle_path = Path(raw_path)
            if not puzzle_path.is_absolute():
                puzzle_path = Path.cwd() / puzzle_path
            if not puzzle_path.is_file():
                data["issues"].append(f"{instance_id}: puzzle file missing: {raw_path}")
                continue
            content = "".join(ch for ch in puzzle_path.read_text(encoding="ascii") if not ch.isspace())
            if len(content) != int(group[0]["grid_size"]) ** 2:
                data["issues"].append(f"{instance_id}: puzzle character count does not equal grid_size squared")
            clues = sum(ch not in ".0_" for ch in content)
            if clues != int(group[0]["clue_count"]):
                data["issues"].append(f"{instance_id}: puzzle clues {clues} != metadata {group[0]['clue_count']}")
            n = int(group[0]["grid_size"])
            for pos, ch in enumerate(content):
                if ch not in ".0_" and (ch.lower() not in "0123456789abcdefghijklmnopqrstuvwxyz" or int(ch, 36) < 1 or int(ch, 36) > n):
                    data["issues"].append(f"{instance_id}: invalid puzzle symbol {ch!r} at cell {pos}")
    data.update(status_counts=status_counts, backend_counts=backend_counts, instance_groups=ids)


def make_summary(dataset):
    rows = dataset["rows"]
    groups = defaultdict(list)
    for row in rows:
        key = (row["grid_size"], row["backend"], row["amo_method"] or "(not applicable)", row["solver"])
        groups[key].append(row)
    result = []
    for (grid_size, backend, amo_method, solver), group in sorted(groups.items(), key=lambda item: (int(item[0][0]), item[0][1], item[0][2], item[0][3])):
        status = Counter(r["status"] for r in group)
        solve = [number(r["t_solve"]) for r in group if r["t_solve"] != ""]
        encode = [number(r["t_encode"]) for r in group if r["t_encode"] != ""]
        nvars = [number(r["n_vars"]) for r in group if r["n_vars"] != ""]
        clauses = [number(r["n_clauses"]) for r in group if r["n_clauses"] != ""]
        sm, si = median_iqr(solve)
        em, ei = median_iqr(encode)
        limits = sorted({r["time_limit_s"] for r in group})
        result.append({
            "dataset": dataset["label"], "grid_size": grid_size, "backend": backend,
            "amo_method": amo_method, "solver": solver, "n_rows": len(group),
            "n_instances": len({r["instance_id"] for r in group}),
            "n_solved": status["SOLVED"], "n_unsat": status["UNSAT"],
            "n_timeout": status["TIMEOUT"], "n_unknown": status["UNKNOWN"],
            "n_backend_unavailable": status["BACKEND_UNAVAILABLE"],
            "n_invalid": status["INVALID_SOLUTION"], "n_error": status["ERROR"],
            "solve_median_s": sm, "solve_iqr_s": si, "encode_median_s": em,
            "encode_iqr_s": ei,
            "n_vars_median": f"{statistics.median(nvars):.1f}" if nvars else "",
            "n_clauses_median": f"{statistics.median(clauses):.1f}" if clauses else "",
            "time_limit_s": ";".join(limits),
        })
    return result


def fmt(value, digits=6):
    return "—" if value == "" else f"{float(value):.{digits}f}"


def write_svg(full_summary, target):
    rows = [r for r in full_summary if r["backend"] == "pysat" and r["encode_median_s"] != ""]
    width, height = 940, 430
    left, right, top, bottom = 230, 40, 40, 55
    methods = list(AMO_METHODS)
    sizes = sorted({r["grid_size"] for r in rows}, key=int)
    maxv = max((float(r["encode_median_s"]) for r in rows), default=1.0) or 1.0
    bar_w = 520
    colors = {"9": "#3977b8", "16": "#e08b32"}
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<text x="24" y="25" font-family="sans-serif" font-size="17" font-weight="bold">Full run: median SAT encoding time by AMO and grid size</text>']
    chart_top, chart_bottom = 55, height-bottom
    for i in range(6):
        value = maxv * i / 5
        x = left + bar_w * i / 5
        lines.append(f'<line x1="{x:.1f}" y1="{chart_top}" x2="{x:.1f}" y2="{chart_bottom}" stroke="#ddd"/>')
        lines.append(f'<text x="{x:.1f}" y="{height-26}" text-anchor="middle" font-family="sans-serif" font-size="11">{value:.4f}</text>')
    for mi, method in enumerate(methods):
        y = chart_top + 14 + mi * 62
        lines.append(f'<text x="{left-12}" y="{y+10}" text-anchor="end" font-family="sans-serif" font-size="13">{method}</text>')
        for si, size in enumerate(sizes):
            row = next((r for r in rows if r["amo_method"] == method and r["grid_size"] == size), None)
            if row is None:
                continue
            val = float(row["encode_median_s"])
            yy = y + si * 22
            ww = bar_w * val / maxv
            lines.append(f'<rect x="{left}" y="{yy}" width="{ww:.2f}" height="15" fill="{colors.get(size, "#6b8e23")}"/>')
            lines.append(f'<text x="{left+ww+5:.1f}" y="{yy+12}" font-family="sans-serif" font-size="10">{val:.6f}s (n={row["n_rows"]})</text>')
    legend_y = height - 7
    lx = left
    for size in sizes:
        lines.append(f'<rect x="{lx}" y="{legend_y-11}" width="12" height="12" fill="{colors.get(size, "#6b8e23")}"/>')
        lines.append(f'<text x="{lx+17}" y="{legend_y}" font-family="sans-serif" font-size="11">{size}x{size}</text>')
        lx += 90
    lines.append(f'<text x="{left+bar_w/2}" y="{height-5}" text-anchor="middle" font-family="sans-serif" font-size="10">seconds</text>')
    lines.append('</svg>')
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(v) for v in row) + " |" for row in rows]
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, default=Path("results/benchmark_raw_20261017T145314Z-pilot.csv"))
    parser.add_argument("--full", type=Path, default=Path("results/benchmark_raw_20261017T145326Z-full.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/reports"))
    args = parser.parse_args()
    datasets = [read_dataset("pilot", args.pilot), read_dataset("full", args.full)]
    for data in datasets:
        audit_dataset(data)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = [row for d in datasets for row in make_summary(d)]
    with (args.output_dir / "benchmark_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary)
    full_rows = make_summary(datasets[1])
    write_svg(full_rows, args.output_dir / "full_encoding_time.svg")

    report = ["# Sudoku SAT benchmark analysis", "",
              "This report is regenerated from the raw CSVs listed below. Raw CSV files were read only.", "",
              "## Input files", ""]
    for d in datasets:
        report.append(f"- `{d['path']}`: {len(d['rows'])} data rows; {len(d['instance_groups'])} instances.")
    report += ["", "## Data audit", "",
               "The CSV schema includes the required core fields: instance and seed metadata, AMO, solver/backend, status, solved/timeout/time limit, dimensions, encoding/solve timings, and version. `n_constraints`, memory metrics, keep ratio, and error are additional fields.", ""]
    for d in datasets:
        report.append(f"### {d['label']}")
        report.append("")
        report.append(f"- Rows: {len(d['rows'])}; unique instance IDs: {len(d['instance_groups'])}; duplicate configuration keys: {len(d['duplicates'])}.")
        report.append(f"- Backends: {dict(d['backend_counts'])}.")
        report.append(f"- Statuses: {dict(d['status_counts'])}.")
        report.append(f"- Limit(s): {', '.join(sorted({r['time_limit_s'] for r in d['rows']}))} seconds.")
        report.append(f"- Audit findings: {len(d['issues'])}.")
        if d["issues"]:
            report.extend(f"  - {issue}" for issue in d["issues"])
        else:
            report.append("- No schema, per-instance metadata, puzzle-file/clue-count, or status consistency errors detected.")
        report.append("")
    report += ["### Cross-run fairness", "",
               "Pilot and full CSVs are reported separately. They share the b3/seed17 and b4/seed17 instance IDs, so pooling them as independent observations would double-count those inputs. Within each CSV, SAT AMO rows and CP-SAT rows use the same instance metadata and paired puzzle path. The full run has three seeds per size; pilot has one. Time limits differ (pilot 10 s, full 30 s), so no cross-run runtime comparison is made.", "",
               "## Results", "",
               "All observed rows have status `BACKEND_UNAVAILABLE`: Python-SAT cannot import `pysat` and CP-SAT cannot import `ortools`. Therefore there are no solved, UNSAT, timeout, or unknown observations. Backend unavailability is not an UNSAT result and is not evidence about solver performance.", "",
               "Encoding measurements exist for SAT AMO configurations; solve-time values are blank. The table below reports full-run medians across three instances per grid size, with `n` shown. IQR uses linearly interpolated inclusive quartiles; with three instances it is Q3−Q1. CP-SAT has no encoding metrics in the raw data because its backend was unavailable.", ""]
    tab = []
    for r in full_rows:
        tab.append([r["grid_size"], r["backend"], r["amo_method"], r["solver"], r["n_rows"], r["n_instances"],
                    r["n_solved"], r["n_timeout"], r["n_unknown"], r["n_unsat"],
                    fmt(r["encode_median_s"]), fmt(r["encode_iqr_s"]), fmt(r["solve_median_s"]), fmt(r["solve_iqr_s"]),
                    fmt(r["n_vars_median"], 1), fmt(r["n_clauses_median"], 1)])
    report.append(md_table(["Grid", "Backend", "AMO", "Solver", "Rows", "n instances", "Solved", "Timeout", "Unknown", "UNSAT", "Encode median s", "Encode IQR s", "Solve median s", "Solve IQR s", "Median vars", "Median clauses"], tab))
    report += ["", "### Per-instance outcomes (full run)", "",
               "Each line below is a paired puzzle group. No rows were removed from these counts.", ""]
    instance_table = []
    for instance_id, group in sorted(datasets[1]["instance_groups"].items()):
        first = group[0]
        counts = Counter(r["status"] for r in group)
        instance_table.append([instance_id, first["grid_size"], first["seed"], first["clue_count"], len(group),
                               counts["SOLVED"], counts["TIMEOUT"], counts["UNKNOWN"], counts["UNSAT"], counts["BACKEND_UNAVAILABLE"]])
    report.append(md_table(["Instance", "Grid", "Seed", "Clues", "Rows", "Solved", "Timeout", "Unknown", "UNSAT", "Backend unavailable"], instance_table))
    report += ["", "No SAT-vs-CP-SAT runtime conclusion can be drawn because neither backend ran. Likewise, no solver or AMO solve-speed ranking is supported. Encoding-time and formula-size summaries describe the available encoding rows only; they are not end-to-end solve comparisons.", "",
               "## Limitations", "",
               "- Puzzles were generated by puncturing seeded solved grids at keep ratio 0.35; uniqueness is not guaranteed.",
               "- The full dataset contains only three instances per size (the pilot contains one per size).",
               "- No solving occurred due to missing Python-SAT/PySAT and OR-Tools packages; the configured limits were 10 s (pilot) and 30 s (full), but they were not consumed by solver runs.",
               "- Raw version metadata identifies Python 3.14.4 and marks unavailable solver packages. Hardware/OS details are not recorded in these CSVs, so machine-level reproducibility is limited.",
               "- The chart shows full-run SAT encoding-time medians only; it does not imply solver performance.", "",
               "## Generated artifacts", "",
               "- `benchmark_summary.csv`: per dataset, grid size, backend, AMO, and solver; missing measurements stay blank.",
               "- `full_encoding_time.svg`: full-run SAT encoding-time chart.",
               "- `benchmark_analysis.md`: this report.", "",
               "## Data Availability", "",
               "Source code, sample input, raw benchmark CSVs, paired puzzle instances, and generated summaries are available at https://github.com/canknotcaw/sudoku_sat_solver.", ""]
    (args.output_dir / "benchmark_analysis.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote {args.output_dir / 'benchmark_summary.csv'} ({len(summary)} grouped rows)")
    print(f"Wrote {args.output_dir / 'full_encoding_time.svg'}")
    print(f"Wrote {args.output_dir / 'benchmark_analysis.md'}")
    for d in datasets:
        print(f"{d['label']}: rows={len(d['rows'])}, instances={len(d['instance_groups'])}, statuses={dict(d['status_counts'])}, issues={len(d['issues'])}")


if __name__ == "__main__":
    main()
