import builtins
from pathlib import Path

import pytest

from novelentitymatcher.ingestion import cli
from novelentitymatcher.ingestion.base import IngestionFailure, IngestionRunResult


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


def test_cli_all_concurrent_continue_on_error_reports_structured_failures(
    monkeypatch, capsys
):
    class _DummyFetcher:
        def __init__(self, raw_dir, processed_dir):
            self.raw_dir = raw_dir
            self.processed_dir = processed_dir

    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if not name.startswith("novelentitymatcher.ingestion."):
            return original_import(name, globals, locals, fromlist, level)

        class _M:
            LanguagesFetcher = _DummyFetcher
            CurrenciesFetcher = _DummyFetcher

        return _M

    def fake_run_all_concurrent_detailed(fetchers, **kwargs):
        assert kwargs["continue_on_error"] is True
        return IngestionRunResult(
            output_paths=[Path("/tmp/languages.csv")],
            failures=[
                IngestionFailure(
                    fetcher="CurrenciesFetcher",
                    output_filename="currencies.csv",
                    error_type="RuntimeError",
                    message="boom",
                )
            ],
        )

    monkeypatch.setattr(
        cli,
        "INGESTORS",
        {
            "languages": object(),
            "currencies": object(),
            "all": None,
        },
    )
    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(
        cli, "run_all_concurrent_detailed", fake_run_all_concurrent_detailed
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["all", "--concurrent", "--continue-on-error"])

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "Concurrent ingestion failures:" in captured.err
    assert "CurrenciesFetcher (currencies.csv): RuntimeError: boom" in captured.err
