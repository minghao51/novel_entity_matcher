from pathlib import Path

import pytest

from novelentitymatcher.ingestion import cli


def test_cli_lists_datasets(capsys):
    cli.main(["--list"])

    out = capsys.readouterr().out
    assert "Available datasets:" in out
    assert "languages" in out
    assert "occupations" in out
    assert "all" in out


def test_cli_forwards_base_dirs(monkeypatch, tmp_path):
    calls = []

    def fake_ingestor(raw_dir=None, processed_dir=None):
        calls.append((raw_dir, processed_dir))

    monkeypatch.setitem(cli.INGESTORS, "languages", fake_ingestor)

    raw_base = tmp_path / "raw-base"
    processed_base = tmp_path / "processed-base"
    cli.main(
        [
            "languages",
            "--raw-dir",
            str(raw_base),
            "--processed-dir",
            str(processed_base),
        ]
    )

    assert calls == [(Path(raw_base), Path(processed_base))]


def test_cli_all_exits_non_zero_on_failure(monkeypatch, capsys):
    calls = []

    def ok_ingestor(raw_dir=None, processed_dir=None):
        calls.append("ok")

    def failing_ingestor(raw_dir=None, processed_dir=None):
        calls.append("fail")
        raise RuntimeError("boom")

    monkeypatch.setattr(
        cli,
        "INGESTORS",
        {
            "languages": ok_ingestor,
            "currencies": failing_ingestor,
            "all": None,
        },
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["all"])
    assert exc_info.value.code == 1

    assert calls == ["ok", "fail"]

    captured = capsys.readouterr()
    assert "Error ingesting currencies: boom" in captured.err
    assert "Ingestion completed with failures:" in captured.err
