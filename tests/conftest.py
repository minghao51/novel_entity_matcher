import pytest

from novelentitymatcher.utils.embeddings import get_default_cache


@pytest.fixture(autouse=True)
def clear_model_cache():
    """Clear the global model cache before each test."""
    cache = get_default_cache()
    cache.clear()
    yield
    cache.clear()


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "asyncio: compatibility alias for async tests executed via anyio",
    )


def pytest_collection_modifyitems(items):
    for item in items:
        if "asyncio" in item.keywords and "anyio" not in item.keywords:
            item.add_marker(pytest.mark.anyio)

        nodeid = item.nodeid
        if "/unit/" in nodeid and not any(
            m.name == "unit" for m in item.iter_markers()
        ):
            item.add_marker(pytest.mark.unit)
        elif "/integration/" in nodeid:
            if not any(m.name == "integration" for m in item.iter_markers()):
                item.add_marker(pytest.mark.integration)
            if not any(m.name == "slow" for m in item.iter_markers()):
                item.add_marker(pytest.mark.slow)
