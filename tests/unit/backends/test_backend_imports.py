import pytest

from novelentitymatcher.backends import HFReranker, HFEmbedding
from novelentitymatcher.backends.sentencetransformer import (
    HFReranker as NewHFReranker,
    HFEmbedding as NewHFEmbedding,
)


def test_backend_exports_use_canonical_module():
    assert HFEmbedding is NewHFEmbedding
    assert HFReranker is NewHFReranker


def test_misspelled_backend_module_is_removed():
    with pytest.raises(ModuleNotFoundError):
        __import__(
            "novelentitymatcher.backends.sentencetranformer", fromlist=["HFEmbedding"]
        )
