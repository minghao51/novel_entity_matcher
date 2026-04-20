from novelentitymatcher.ingestion.timezones import WorldTimeAPIFetcher


def test_worldtimeapi_uses_https():
    assert WorldTimeAPIFetcher.TZ_LIST_URL.startswith("https://")
