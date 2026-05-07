"""
GEPA optimizer for DSPy-based class proposals.

Uses ProposalReviewManager approved/rejected records as training signal
to optimize the DSPy proposal module via GEPA teleprompter.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib

try:
    import dspy
    from dspy.teleprompt import GEPA

    _DSPY_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    _DSPY_AVAILABLE = False
    dspy = None
    GEPA = None

from .dspy_module import (
    DSPyProposalModule,
    proposal_metric,
    review_records_to_examples,
)

logger = logging.getLogger(__name__)


class DSPyProposalOptimizer:
    """Optimize DSPy proposal modules using GEPA and review record training data.

    Reads approved/rejected proposals from ProposalReviewManager,
    converts them to DSPy training examples, and runs GEPA optimization
    to refine the proposal module's instructions and few-shot examples.

    Args:
        lm: Optional DSPy LM instance. If not provided, uses global dspy.configure().
        module: Optional pre-configured DSPyProposalModule. Creates a new one if not provided.
        save_path: Optional path to save/load optimized modules.
        auto_mode: GEPA auto mode ('light', 'medium', or 'heavy').
    """

    def __init__(
        self,
        lm: dspy.LM | None = None,
        module: DSPyProposalModule | None = None,
        save_path: str | Path | None = None,
        auto_mode: str = "light",
    ):
        if not _DSPY_AVAILABLE:
            raise ImportError(
                "dspy is required for DSPyProposalOptimizer. "
                "Install with: pip install 'novel-entity-matcher[opinion]'"
            )

        self.lm = lm
        if lm is not None:
            dspy.configure(lm=lm)
        self.module = module or DSPyProposalModule(lm=lm)
        self.save_path = Path(save_path) if save_path else None
        self.auto_mode = auto_mode
        self._optimized: DSPyProposalModule | None = None

    def compile(
        self,
        approved_records: list[dict[str, Any]],
        rejected_records: list[dict[str, Any]] | None = None,
        max_evals: int = 20,
        save: bool = True,
    ) -> DSPyProposalModule:
        """Compile/optimize the proposal module using GEPA.

        Args:
            approved_records: List of approved ProposalReviewRecord dicts (positive training data).
            rejected_records: List of rejected ProposalReviewRecord dicts (negative training data).
            max_evals: Maximum number of full evaluations for GEPA.
            save: Whether to save the optimized module to save_path.

        Returns:
            Optimized DSPyProposalModule.
        """
        if not approved_records:
            logger.warning("No approved records provided — skipping optimization")
            self._optimized = self.module
            return self.module

        trainset = review_records_to_examples(
            approved_records=approved_records,
            rejected_records=rejected_records,
        )

        logger.info(
            "Starting GEPA optimization with %d training examples (auto=%s, max_evals=%d)",
            len(trainset),
            self.auto_mode,
            max_evals,
        )

        optimizer = GEPA(
            metric=proposal_metric,
            auto=self.auto_mode,
            max_full_evals=max_evals,
            num_threads=1,
        )

        optimized_module = optimizer.compile(
            self.module,
            trainset=trainset,
        )

        self._optimized = optimized_module

        if save and self.save_path:
            self._save(optimized_module)

        logger.info("GEPA optimization completed")
        return optimized_module

    def load(self, path: str | Path | None = None) -> DSPyProposalModule:
        """Load a previously optimized module from disk.

        Args:
            path: Path to load from. Defaults to save_path from constructor.

        Returns:
            Loaded DSPyProposalModule.
        """
        load_path = Path(path) if path else self.save_path
        if not load_path or not load_path.exists():
            raise FileNotFoundError(f"No saved optimizer found at {load_path}")

        raw = json.loads(load_path.read_text())
        module_pickle = raw.get("module_pickle", "optimized_module.pkl")
        module_path = load_path.parent / module_pickle
        if not module_path.exists():
            raise FileNotFoundError(
                f"Optimized module payload not found at {module_path}"
            )
        module = joblib.load(module_path)
        if raw.get("optimized_module"):
            logger.info("Loaded previously optimized module from %s", load_path)
        self._optimized = module
        return module

    def _save(self, module: DSPyProposalModule) -> None:
        """Save the optimized module metadata to disk."""
        if not self.save_path:
            return
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        module_pickle = "optimized_module.pkl"
        module_path = self.save_path.parent / module_pickle
        joblib.dump(module, module_path)
        payload = {
            "optimized_module": True,
            "auto_mode": self.auto_mode,
            "module_type": "DSPyProposalModule",
            "module_pickle": module_pickle,
        }
        self.save_path.write_text(json.dumps(payload, indent=2))

    @property
    def optimized_module(self) -> DSPyProposalModule | None:
        """Get the optimized module (None if compile() hasn't been called)."""
        return self._optimized
