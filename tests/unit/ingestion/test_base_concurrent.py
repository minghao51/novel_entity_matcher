import asyncio
from pathlib import Path

import pytest

from novelentitymatcher.ingestion.base import (
    BaseFetcher,
    run_all_concurrent_detailed,
    run_concurrent_detailed,
)


class _OkFetcher(BaseFetcher):
    def fetch(self):
        return [{"id": "1", "name": "ok"}]

    def process(self, raw_data):
        return raw_data


class _FailFetcher(BaseFetcher):
    def fetch(self):
        raise RuntimeError("fetch-failed")

    def process(self, raw_data):
        return raw_data


def test_run_concurrent_detailed_continue_on_error(tmp_path):
    ok = _OkFetcher(tmp_path / "raw_ok", tmp_path / "processed_ok")
    bad = _FailFetcher(tmp_path / "raw_bad", tmp_path / "processed_bad")
    fetchers = [(ok, "ok.csv"), (bad, "bad.csv")]

    result = asyncio.run(
        run_concurrent_detailed(fetchers, continue_on_error=True, max_concurrent=2)
    )

    assert len(result.output_paths) == 1
    assert isinstance(result.output_paths[0], Path)
    assert len(result.failures) == 1
    failure = result.failures[0]
    assert failure.fetcher == "_FailFetcher"
    assert failure.output_filename == "bad.csv"
    assert failure.error_type == "RuntimeError"
    assert "fetch-failed" in failure.message


def test_run_all_concurrent_detailed_fail_fast(tmp_path):
    bad = _FailFetcher(tmp_path / "raw_bad", tmp_path / "processed_bad")
    fetchers = [(bad, "bad.csv")]

    with pytest.raises(RuntimeError, match="fetch-failed"):
        run_all_concurrent_detailed(fetchers, continue_on_error=False, max_concurrent=1)
