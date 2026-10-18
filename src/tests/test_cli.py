import pytest

from src.main import main


@pytest.mark.parametrize("unique", [False, True])
def test_cli_reports_invalid_pysat_solver_without_traceback(tmp_path, capsys, unique):
    puzzle = tmp_path / "empty.suk"
    puzzle.write_text("." * 81, encoding="utf-8")
    argv = [str(puzzle), "--solver", "not_a_real_pysat_solver"]
    if unique:
        argv.append("--check-unique")

    with pytest.raises(SystemExit) as exc_info:
        main(argv)

    assert exc_info.value.code == 2
    err = capsys.readouterr().err
    assert "unknown or unavailable PySAT solver 'not_a_real_pysat_solver'" in err
    assert "recognized names include:" in err
    assert "Traceback" not in err


def test_cli_missing_puzzle_has_argument_error(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc_info:
        main([str(tmp_path / "missing.suk")])
    assert exc_info.value.code == 2
    err = capsys.readouterr().err
    assert "cannot read puzzle file" in err
    assert "Traceback" not in err


def test_cli_stream_reports_separate_encoding_and_solver_load(tmp_path, capsys):
    puzzle = tmp_path / "empty-4x4.suk"
    puzzle.write_text("." * 16, encoding="utf-8")
    main([str(puzzle), "--box", "2", "--amo", "sequential", "--stream"])
    out = capsys.readouterr().out
    assert "solver_load=" in out
    assert "encode=" in out
    assert "clauses=" in out and "vars=" in out
    assert "UNSATISFIABLE" not in out


def test_cli_rejects_stream_with_uniqueness(tmp_path, capsys):
    puzzle = tmp_path / "empty-4x4.suk"
    puzzle.write_text("." * 16, encoding="utf-8")
    with pytest.raises(SystemExit) as exc_info:
        main([str(puzzle), "--box", "2", "--stream", "--check-unique"])
    assert exc_info.value.code == 2
    assert "cannot be combined with --check-unique" in capsys.readouterr().err
