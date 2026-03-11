from pathlib import Path

from semanticmatcher.ingestion import cli


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
