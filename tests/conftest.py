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
    """Support asyncio-marked tests when pytest-asyncio is not installed."""
    config.addinivalue_line(
        "markers",
        "asyncio: compatibility alias for async tests executed via anyio",
    )


def pytest_collection_modifyitems(items):
    """Map legacy asyncio markers onto anyio so async tests still execute."""
    for item in items:
        if "asyncio" in item.keywords and "anyio" not in item.keywords:
            item.add_marker(pytest.mark.anyio)
