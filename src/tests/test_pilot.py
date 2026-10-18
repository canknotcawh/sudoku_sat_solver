import csv

from src import pilot


def test_pilot_schema_has_explicit_status_rss_and_limits():
    required = {"schema_version", "status", "peak_rss_mb", "rss_available", "memory_limit_mb",
                "timeout_s", "t_encode", "t_solve", "t_total", "t_wall", "puzzle_sha256"}
    assert required <= set(pilot.PILOT_FIELDS)
    assert pilot.PILOT_STATUSES == {"solved", "timeout", "memory_limit", "encoding_only", "error"}
    assert len(pilot.PILOT_FIELDS) == len(set(pilot.PILOT_FIELDS))


def test_job_expansion_is_explicit_and_encode_only_has_no_solver_product():
    jobs = pilot.build_pilot_jobs(7, 2, [41, 42], ["product", "sequential"],
                                  ["cadical195", "glucose42"], False)
    assert len(jobs) == 8
    assert {job["solver"] for job in jobs} == {"cadical195", "glucose42"}
    assert {job["seed"] for job in jobs} == {41, 42}
    assert jobs[0]["givens"] == next(job["givens"] for job in jobs if job["puzzle"] == 0)

    encode_jobs = pilot.build_pilot_jobs(8, 1, [99], ["product"], ["cadical195"], True)
    assert len(encode_jobs) == 1
    assert encode_jobs[0]["solver"] == "-"
    assert encode_jobs[0]["encode_only"]


def test_worker_only_reports_solved_after_independent_validator(monkeypatch):
    class FakeEncoder:
        def __init__(self, **_kwargs):
            self.box = 1

        def encode(self, _givens):
            return [[1]], 1

    monkeypatch.setattr(pilot, "SudokuEncoder", FakeEncoder)
    monkeypatch.setattr(pilot, "solve", lambda *_args, **_kwargs: [[1]])
    monkeypatch.setattr(pilot, "is_valid_solution", lambda *_args: False)
    # Exercise worker over a one-way pipe, without starting a benchmark job.
    import multiprocessing as mp
    parent, child = mp.Pipe(duplex=False)
    pilot._worker(child, dict(box=1, encoding="product", solver="cadical195", givens=[], encode_only=False), None)
    result = parent.recv()
    assert result["status"] == "error"
    assert result["valid"] is False


def test_run_pilot_job_records_encode_only_and_rss_availability():
    job = dict(box=1, encoding="product", solver="-", givens=[], encode_only=True)
    result = pilot.run_pilot_job(job, timeout_s=5)
    assert result["status"] == "encoding_only"
    assert result["t_encode"] >= 0
    assert isinstance(result["rss_available"], bool)
    assert (result["peak_rss_mb"] != "") == result["rss_available"]
