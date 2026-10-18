"""Tests for benchmark provenance, schema handling, and resume identity."""
import csv
import json
from types import SimpleNamespace

from src import benchmark
from src.benchmark_schema import FIELDS, LEGACY_FIELDS, V2_FIELDS, V3_FIELDS


def test_job_key_changes_with_timeouts_and_configuration():
    job = dict(experiment="sat", method="product", solver="cadical195", unit_alo=True,
               box=3, puzzle=0, keep_ratio=0.4, time_limit=60, worker_timeout=65)
    original = benchmark.job_key(job)
    changed_solver_limit = dict(job, time_limit=120)
    changed_worker_limit = dict(job, worker_timeout=125)
    changed_seed = dict(job, seed=1)
    changed_hash = dict(job, puzzle_sha256="a" * 64)
    assert original != benchmark.job_key(changed_solver_limit)
    assert original != benchmark.job_key(changed_worker_limit)
    assert original != benchmark.job_key(changed_seed)
    assert original != benchmark.job_key(changed_hash)


def test_worker_preserves_nonzero_build_time_on_solver_timeout(monkeypatch):
    import time

    def solver_timeout(_job):
        time.sleep(.02)
        return dict(status="timeout", valid="", t_encode=.01, t_solve=.01,
                    message="solver limit reached")

    monkeypatch.setattr(benchmark, "run_baseline", solver_timeout)
    job = dict(family="ilp_cp", solver_timeout_s=.01)
    result = benchmark.run_job(job, timeout=1.0)
    assert result["status"] == "timeout"
    assert result["timeout_kind"] == "solver"
    assert result["t_encode"] == .01
    assert result["t_solve"] == .01
    assert result["t_wall"] >= result["t_encode"]


def test_worker_process_timeout_has_wall_time_but_no_fake_phase_times(monkeypatch):
    import time

    def slow_runner(_job):
        time.sleep(2)
        return dict(status="solved", valid=True)

    monkeypatch.setattr(benchmark, "run_baseline", slow_runner)
    result = benchmark.run_job(dict(family="ilp_cp", solver_timeout_s=1.0), timeout=.1)
    assert result["status"] == "timeout"
    assert result["timeout_kind"] == "process"
    assert result["t_wall"] < 1.0
    assert "t_solve" not in result
    assert "t_encode" not in result


def test_worker_crash_is_not_recorded_as_timeout(monkeypatch):
    def crash(_job):
        raise SystemExit(7)

    monkeypatch.setattr(benchmark, "run_baseline", crash)
    result = benchmark.run_job(dict(family="ilp_cp", solver_timeout_s=1.0), timeout=1.0)
    assert result["status"] == "crashed"
    assert result["timeout_kind"] == "worker_crash"
    assert result["t_wall"] >= 0


def test_plot_uses_observed_wall_time_and_does_not_cap_at_solver_limit(tmp_path):
    from src.plot_results import agg, load

    path = tmp_path / "timings.csv"
    rows = [
        dict(schema_version=4, solver_timeout_s=60, process_timeout_s=90, timeout_kind="solver",
             t_wall=63.4, t_encode=3.0, t_solve=60.1, t_total=63.1, status="timeout", valid="",
             experiment="baseline", method="cbc_ilp", solver="cbc_ilp", unit_alo="", box=3,
             n=9, puzzle=0, keep_ratio=.4, run_id="r", config_hash="c", seed=1,
             puzzle_sha256="a" * 64, split="evaluation"),
        dict(schema_version=4, solver_timeout_s="", process_timeout_s=65, timeout_kind="process",
             t_wall=65.0, t_encode="", t_solve="", t_total="", status="timeout", valid="",
             experiment="sat", method="product", solver="cadical195", unit_alo=True, box=3,
             n=9, puzzle=1, keep_ratio=.4, run_id="r", config_hash="c", seed=2,
             puzzle_sha256="b" * 64, split="evaluation"),
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    data = load(str(path))
    baseline = data[data.method == "cbc_ilp"]
    summary = agg(baseline, ["n"]).iloc[0]
    assert baseline.iloc[0].time_observed == 63.4
    assert summary["median"] != summary["median"]
    assert summary.median_bound == 63.4
    process_row = data[data.method == "product"].iloc[0]
    assert process_row.time_observed == 65.0
    assert process_row.t_total != 60.0


def test_load_done_only_resumes_matching_v4_run_and_config(tmp_path):
    path = tmp_path / "runs.csv"
    base = dict(schema_version=4, run_id="run-a", config_hash="config-a", git_commit="abc",
                timeout_s=60, worker_timeout_s=65, seed=3000, puzzle_sha256="f" * 64,
                solver_timeout_s="", process_timeout_s=65, timeout_kind="none", t_wall=.31,
                experiment="sat", family="sat", method="product", solver="cadical195",
                unit_alo=True, box=3, n=9, puzzle=0, keep_ratio=0.4,
                n_vars=1, n_clauses=1, t_encode=0.1, t_solve=0.2, t_total=0.3,
                status="solved", valid=True, message="")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerow(base)
        writer.writerow(dict(base, run_id="run-b", config_hash="config-b"))

    done = benchmark.load_done(str(path), "run-a", "config-a")
    assert len(done) == 1
    assert next(iter(done))[-5:] == (None, 65.0, 3000, "f" * 64, "")
    assert benchmark.load_done(str(path), "run-a", "different-config") == set()


def test_legacy_csv_is_preserved_and_routes_to_v4_sidecar(tmp_path):
    legacy = tmp_path / "results.csv"
    original = ",".join(LEGACY_FIELDS) + "\n" + "legacy,data,stays,intact\n"
    legacy.write_text(original, encoding="utf-8")

    target, detected = benchmark.resolve_csv_path(str(legacy))

    assert detected is True
    assert target == str(tmp_path / "results.v4.csv")
    assert legacy.read_text(encoding="utf-8") == original
    assert benchmark.load_done(str(legacy), "new-run", "new-config") == set()


def test_v2_csv_routes_to_v4_sidecar_and_is_preserved(tmp_path):
    path = tmp_path / "old.csv"
    original = ",".join(V2_FIELDS) + "\nold,data\n"
    path.write_text(original, encoding="utf-8")
    target, detected = benchmark.resolve_csv_path(str(path))
    assert detected is True
    assert target == str(tmp_path / "old.v4.csv")
    assert path.read_text(encoding="utf-8") == original


def test_v3_csv_routes_to_v4_sidecar_and_is_preserved(tmp_path):
    path = tmp_path / "old-v3.csv"
    original = ",".join(V3_FIELDS) + "\nold,data\n"
    path.write_text(original, encoding="utf-8")
    target, detected = benchmark.resolve_csv_path(str(path))
    assert detected is True
    assert target == str(tmp_path / "old-v3.v4.csv")
    assert path.read_text(encoding="utf-8") == original


def test_config_hash_changes_when_seed_density_or_timeout_changes(monkeypatch):
    monkeypatch.setattr(benchmark, "git_commit", lambda: "commit")
    monkeypatch.setattr(benchmark, "source_fingerprint", lambda: "source")
    monkeypatch.setattr(benchmark, "package_versions", lambda: {})
    base = SimpleNamespace(seeds=[1, 2, 3], keep_ratios=[.3, .5], tune_count=1, timeout=60,
                           out="results.csv", new_run=False)
    cfg = benchmark.canonical_config(base, [], {}, None)
    variants = [SimpleNamespace(**{**vars(base), "seeds": [1, 2, 4]}),
                SimpleNamespace(**{**vars(base), "keep_ratios": [.3, .6]}),
                SimpleNamespace(**{**vars(base), "timeout": 120})]
    for changed in variants:
        assert benchmark.config_hash(cfg) != benchmark.config_hash(
            benchmark.canonical_config(changed, [], {}, None))


def test_unknown_csv_schema_is_not_modified(tmp_path):
    path = tmp_path / "unknown.csv"
    original = "unexpected,columns\n1,2\n"
    path.write_text(original, encoding="utf-8")
    try:
        benchmark.resolve_csv_path(str(path))
    except SystemExit as exc:
        assert "unrecognized schema" in str(exc)
    else:
        raise AssertionError("unknown CSV schema should be rejected")
    assert path.read_text(encoding="utf-8") == original


def test_size_only_instances_include_size_boxes_and_full_hash(tmp_path):
    args = SimpleNamespace(boxes=[3], puzzles=1, size_boxes=[7, 8], only=["size"], keep_ratio=0.4)

    instances = benchmark.save_instances(args, str(tmp_path))

    by_size = {record["n"]: record for record in instances}
    assert {9, 49, 64}.issubset(by_size)
    for n in (49, 64):
        rec = by_size[n]
        assert rec["seed"] == rec["box"] * 1000
        assert len(rec["sha256"]) == 64
        assert rec["file"] is None


def test_metadata_records_config_hash_and_schema(tmp_path, monkeypatch):
    args = SimpleNamespace(boxes=[3], puzzles=1, size_boxes=[7], keep_ratio=0.4,
                           timeout=60, only=["size"], out="unused.csv", new_run=False)
    monkeypatch.setattr(benchmark, "git_commit", lambda: "abc123")
    monkeypatch.setattr(benchmark, "source_fingerprint", lambda: "src123")
    monkeypatch.setattr(benchmark, "package_versions", lambda: {"python-sat": "test"})
    monkeypatch.setattr(benchmark.platform, "python_version", lambda: "3.x")
    monkeypatch.setattr(benchmark.platform, "platform", lambda: "test-platform")
    monkeypatch.setattr(benchmark.platform, "machine", lambda: "test-machine")
    monkeypatch.setattr(benchmark.os, "cpu_count", lambda: 4)
    cfg = benchmark.canonical_config(args, ["cadical195"], {}, "cadical195")
    cfg_hash = benchmark.config_hash(cfg)
    path = tmp_path / "meta.json"

    benchmark.write_meta(str(path), args, ["cadical195"], {}, "cadical195", [], "run-123", cfg_hash)

    run = json.loads(path.read_text(encoding="utf-8"))["runs"][0]
    assert run["schema_version"] == 4
    assert run["run_id"] == "run-123"
    assert run["config_hash"] == cfg_hash
    assert run["config"]["source_fingerprint"] == "src123"


def test_auditor_accepts_legacy_v1_header(tmp_path):
    from src.plot_results import audit_csv

    path = tmp_path / "legacy.csv"
    with path.open("w", newline="") as stream:
        csv.writer(stream).writerow(LEGACY_FIELDS)

    failures, info = audit_csv(str(path))
    assert failures == []
    assert any("schema v1 ok" in item for item in info)


def test_auditor_accepts_v2_header(tmp_path):
    from src.plot_results import audit_csv

    path = tmp_path / "v2.csv"
    with path.open("w", newline="") as stream:
        csv.writer(stream).writerow(V2_FIELDS)
    failures, info = audit_csv(str(path))
    assert failures == []
    assert any("schema v2 ok" in item for item in info)


def test_v4_audit_links_size_only_row_to_run_metadata(tmp_path, monkeypatch):
    from src.plot_results import run_audit

    args = SimpleNamespace(boxes=[], puzzles=1, size_boxes=[2], only=["size"], keep_ratio=0.4,
                           timeout=60, out="unused.csv", new_run=False)
    instances = benchmark.save_instances(args, str(tmp_path))
    rec = instances[0]
    monkeypatch.setattr(benchmark, "git_commit", lambda: "abc123")
    monkeypatch.setattr(benchmark, "source_fingerprint", lambda: "src123")
    monkeypatch.setattr(benchmark, "package_versions", lambda: {"python-sat": "test"})
    cfg = benchmark.canonical_config(args, ["cadical195"], {}, "cadical195")
    cfg_hash = benchmark.config_hash(cfg)
    run_id = "run-size"
    benchmark.write_meta(str(tmp_path / "meta.json"), args, ["cadical195"], {}, "cadical195",
                         instances, run_id, cfg_hash)

    row = dict(schema_version=4, run_id=run_id, config_hash=cfg_hash, git_commit="abc123",
               timeout_s=60, worker_timeout_s=600, seed=rec["seed"], puzzle_sha256=rec["sha256"],
               solver_timeout_s="", process_timeout_s=600, timeout_kind="none", t_wall=.015,
               split="descriptive", uniqueness_checked=rec["uniqueness_checked"],
               uniqueness_status=rec["uniqueness_status"],
               solution_guarantee=rec["solution_guarantee"],
               experiment="size", family="size", method="sequential", solver="-", unit_alo=True,
               box=2, n=4, puzzle=0, keep_ratio=0.4, n_vars=64, n_clauses=20,
               t_encode=0.01, t_solve=0.0, t_total=0.01, status="encoded", valid="", message="")
    csv_path = tmp_path / "results.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerow(row)

    assert run_audit(str(csv_path), str(tmp_path)) is True


def test_seed_density_jobs_share_exact_puzzle_across_methods():
    args = SimpleNamespace(boxes=[2], puzzles=3, seeds=[41, 99], tune_count=1,
                           keep_ratios=[0.3, 0.5], timeout=7, only=["sat", "baseline"])
    jobs = benchmark.build_jobs(args, ["cadical195"], "cadical195")
    target = [j for j in jobs if j["box"] == 2 and j["seed"] == 41 and j["keep_ratio"] == 0.3]
    assert target
    assert {j["split"] for j in target} == {"tune"}
    assert len({benchmark.puzzle_sha256(j["givens"]) for j in target}) == 1
    assert len({tuple(tuple(g) for g in j["givens"]) for j in target}) == 1
    assert {j["seed"] for j in jobs} == {41, 99}
    assert {j["keep_ratio"] for j in jobs} == {0.3, 0.5}
    assert {j["split"] for j in jobs} == {"tune", "evaluation"}


def test_size_only_instance_records_every_density_and_explicit_seed(tmp_path):
    args = SimpleNamespace(boxes=[], puzzles=3, seeds=[101, 202], size_boxes=[2],
                           keep_ratios=[0.25, 0.45], only=["size"], tune_count=1)
    records = benchmark.save_instances(args, str(tmp_path))
    assert {(r["seed"], r["keep_ratio"]) for r in records} == {(101, .25), (101, .45)}
    assert all(r["uniqueness_checked"] is False and r["uniqueness_status"] == "not_checked" for r in records)
    assert all(r["solution_guarantee"] == "generated_completion" for r in records)


def test_optional_uniqueness_check_is_recorded(tmp_path, monkeypatch):
    from src.solver import UniquenessResult

    monkeypatch.setattr(benchmark, "solve_check_unique",
                        lambda encoder, clauses, solver_name: UniquenessResult([[1]], True, None))
    args = SimpleNamespace(boxes=[], puzzles=1, seeds=[77], size_boxes=[1],
                           keep_ratios=[0.5], only=["size"], tune_count=0,
                           check_uniqueness=True, uniqueness_solver="cadical195")
    record = benchmark.save_instances(args, str(tmp_path))[0]
    assert record["uniqueness_checked"] is True
    assert record["uniqueness_status"] == "unique"
    assert record["solution_guarantee"] == "generated_completion"


def test_best_sat_selection_uses_tune_only_and_reports_holdout():
    import pandas as pd
    from src.plot_results import best_sat

    rows = []
    for split, method, seconds in (("tune", "product", .1), ("tune", "sequential", .2),
                                   ("evaluation", "product", .9), ("evaluation", "sequential", .3)):
        rows.append(dict(experiment="sat", split=split, method=method, solver="cadical195", n=9,
                         status="solved", time_observed=seconds, t_total=seconds, censored=False, valid=True))
    selected, _ = best_sat(pd.DataFrame(rows))
    assert selected.iloc[0].method == "product"
    assert selected.iloc[0].tune_median == .1
    assert selected.iloc[0]["median"] == .9


def test_distribution_summary_groups_by_density_and_split():
    import pandas as pd
    from src.plot_results import distribution_summary

    data = pd.DataFrame([
        dict(experiment="sat", method="product", solver="cadical195", n=9, keep_ratio=.3,
             split="evaluation", status="solved", timeout_kind="none", valid=True, time_observed=1.0),
        dict(experiment="sat", method="product", solver="cadical195", n=9, keep_ratio=.3,
             split="evaluation", status="timeout", timeout_kind="process", valid="", time_observed=5.0),
        dict(experiment="sat", method="product", solver="cadical195", n=9, keep_ratio=.5,
             split="tune", status="solved", timeout_kind="none", valid=True, time_observed=2.0),
    ])
    result = distribution_summary(data)
    assert len(result) == 2
    evaluation = result[result.keep_ratio == .3].iloc[0]
    assert evaluation.total_jobs == 2
    assert evaluation.valid_solutions == 1
    assert evaluation.timeouts == 1
    assert evaluation.timeout_rate_all == .5
