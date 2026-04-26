import pytest

from novelentitymatcher.backends import HFEmbedding, HFReranker
from novelentitymatcher.backends.sentencetransformer import (
    HFEmbedding as NewHFEmbedding,
)
from novelentitymatcher.backends.sentencetransformer import (
    HFReranker as NewHFReranker,
)


def test_backend_exports_use_canonical_module():
    assert HFEmbedding is NewHFEmbedding
    assert HFReranker is NewHFReranker


def test_misspelled_backend_module_is_removed():
    with pytest.raises(ModuleNotFoundError):
        __import__(
            "novelentitymatcher.backends.sentencetranformer", fromlist=["HFEmbedding"]
        )
