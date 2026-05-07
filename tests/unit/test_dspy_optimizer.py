"""Tests for DSPyProposalOptimizer."""

import json
import tempfile
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

import pytest


class TestDSPyProposalOptimizer:
    """Tests for DSPyProposalOptimizer."""

    SAMPLE_APPROVED: ClassVar[dict] = {
        "review_id": "abc",
        "state": "approved",
        "proposal": {
            "name": "Quantum Biology",
            "description": "Quantum + biology",
            "confidence": 0.9,
            "sample_count": 3,
            "example_samples": ["sample1", "sample2"],
            "justification": "coherent",
            "source_cluster_ids": [0],
        },
        "provenance": {
            "cluster_ids": [0],
            "keywords": ["quantum"],
            "diagnostics": {
                "existing_classes": ["physics"],
                "domain_context": "science",
            },
        },
    }

    def test_compile_no_records_logs_warning(self):
        from novelentitymatcher.novelty.proposal.dspy_optimizer import (
            DSPyProposalOptimizer,
        )

        optimizer = DSPyProposalOptimizer()
        result = optimizer.compile(approved_records=[])
        # With no records, should return the base unoptimized module
        assert result is optimizer.module

    def test_compile_with_records(self):
        from novelentitymatcher.novelty.proposal.dspy_optimizer import (
            DSPyProposalOptimizer,
        )

        with patch(
            "novelentitymatcher.novelty.proposal.dspy_optimizer.GEPA"
        ) as MockGEPA:
            mock_opt_instance = MockGEPA.return_value
            mock_opt_instance.compile.return_value = "optimized"

            optimizer = DSPyProposalOptimizer()
            result = optimizer.compile(
                approved_records=[self.SAMPLE_APPROVED],
                max_evals=5,
            )

            assert result == "optimized"
            assert optimizer._optimized == "optimized"
            MockGEPA.assert_called_once()
            assert MockGEPA.call_args.kwargs["max_full_evals"] == 5

    def test_compile_saves_when_path_set(self):
        from novelentitymatcher.novelty.proposal.dspy_optimizer import (
            DSPyProposalOptimizer,
        )

        with tempfile.TemporaryDirectory() as tmp:
            save_path = Path(tmp) / "opt.json"
            with patch(
                "novelentitymatcher.novelty.proposal.dspy_optimizer.GEPA"
            ) as MockGEPA:
                mock_opt = MockGEPA.return_value
                mock_opt.compile.return_value = {"optimized": True}

                optimizer = DSPyProposalOptimizer(save_path=str(save_path))
                optimizer.compile(
                    approved_records=[self.SAMPLE_APPROVED],
                    max_evals=5,
                    save=True,
                )

                assert save_path.exists()
                data = json.loads(save_path.read_text())
                assert data["optimized_module"] is True
                assert data["auto_mode"] == "light"
                assert data["module_pickle"] == "optimized_module.pkl"
                assert (Path(tmp) / "optimized_module.pkl").exists()

    def test_compile_does_not_save_when_save_false(self):
        from novelentitymatcher.novelty.proposal.dspy_optimizer import (
            DSPyProposalOptimizer,
        )

        with tempfile.TemporaryDirectory() as tmp:
            save_path = Path(tmp) / "opt.json"
            with patch(
                "novelentitymatcher.novelty.proposal.dspy_optimizer.GEPA"
            ) as MockGEPA:
                mock_opt = MockGEPA.return_value
                mock_opt.compile.return_value = {"optimized": True}

                optimizer = DSPyProposalOptimizer(save_path=str(save_path))
                optimizer.compile(
                    approved_records=[self.SAMPLE_APPROVED],
                    max_evals=5,
                    save=False,
                )

                assert not save_path.exists()

    def test_load_restores_saved_module_payload(self):
        from novelentitymatcher.novelty.proposal.dspy_optimizer import (
            DSPyProposalOptimizer,
        )

        with tempfile.TemporaryDirectory() as tmp:
            save_path = Path(tmp) / "opt.json"
            with patch(
                "novelentitymatcher.novelty.proposal.dspy_optimizer.GEPA"
            ) as MockGEPA:
                mock_opt = MockGEPA.return_value
                payload = {"optimized": True, "v": 1}
                mock_opt.compile.return_value = payload

                optimizer = DSPyProposalOptimizer(save_path=str(save_path))
                optimizer.compile(
                    approved_records=[self.SAMPLE_APPROVED],
                    max_evals=5,
                    save=True,
                )

                loaded = optimizer.load()
                assert loaded == payload
                assert optimizer.optimized_module == payload

    def test_load_raises_on_missing_path(self):
        from novelentitymatcher.novelty.proposal.dspy_optimizer import (
            DSPyProposalOptimizer,
        )

        optimizer = DSPyProposalOptimizer()
        with pytest.raises(FileNotFoundError, match="No saved optimizer"):
            optimizer.load("/nonexistent/path.json")

    def test_optimized_module_property_none_before_compile(self):
        from novelentitymatcher.novelty.proposal.dspy_optimizer import (
            DSPyProposalOptimizer,
        )

        optimizer = DSPyProposalOptimizer()
        assert optimizer.optimized_module is None

    def test_optimized_module_property_after_compile(self):
        from novelentitymatcher.novelty.proposal.dspy_optimizer import (
            DSPyProposalOptimizer,
        )

        with patch(
            "novelentitymatcher.novelty.proposal.dspy_optimizer.GEPA"
        ) as MockGEPA:
            mock_opt = MockGEPA.return_value

            class FakeModule:
                pass

            mock_opt.compile.return_value = FakeModule()

            optimizer = DSPyProposalOptimizer()
            result = optimizer.compile(approved_records=[self.SAMPLE_APPROVED])
            assert optimizer.optimized_module is result
