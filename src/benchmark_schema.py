"""CSV schema definitions shared by the benchmark writer and result auditor."""

SCHEMA_VERSION = 4

LEGACY_FIELDS = [
    "experiment", "family", "method", "solver", "unit_alo", "box", "n", "puzzle", "keep_ratio",
    "n_vars", "n_clauses", "t_encode", "t_solve", "t_total", "status", "valid", "message",
]

V2_FIELDS = [
    "schema_version", "run_id", "config_hash", "git_commit", "timeout_s", "worker_timeout_s",
    "seed", "puzzle_sha256", *LEGACY_FIELDS,
]

V3_FIELDS = [
    "schema_version", "run_id", "config_hash", "git_commit", "timeout_s", "worker_timeout_s",
    "seed", "puzzle_sha256", "split", "uniqueness_checked", "uniqueness_status", "solution_guarantee",
    *LEGACY_FIELDS,
]

FIELDS = [
    "schema_version", "run_id", "config_hash", "git_commit", "timeout_s", "worker_timeout_s",
    "solver_timeout_s", "process_timeout_s", "timeout_kind", "t_wall",
    "seed", "puzzle_sha256", "split", "uniqueness_checked", "uniqueness_status", "solution_guarantee",
    *LEGACY_FIELDS,
]
