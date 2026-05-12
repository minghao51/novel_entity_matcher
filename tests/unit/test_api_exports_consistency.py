import novelentitymatcher.api as api


def test_api_all_has_no_duplicates():
    assert len(api.__all__) == len(set(api.__all__))


def test_api_all_symbols_are_resolvable():
    missing = [name for name in api.__all__ if not hasattr(api, name)]
    assert missing == []
