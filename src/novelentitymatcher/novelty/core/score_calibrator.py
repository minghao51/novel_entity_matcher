from __future__ import annotations

import numpy as np


class OODScoreCalibrator:
    def __init__(self, method: str = "minmax"):
        self.method = method
        self._stats: dict[str, dict[str, float]] = {}

    def fit(self, strategy_scores: dict[str, np.ndarray]) -> OODScoreCalibrator:
        for strategy_id, scores in strategy_scores.items():
            p5 = (
                float(np.percentile(scores, 5))
                if len(scores) >= 20
                else float(np.min(scores))
            )
            p95 = (
                float(np.percentile(scores, 95))
                if len(scores) >= 20
                else float(np.max(scores))
            )
            self._stats[strategy_id] = {
                "min": float(np.min(scores)),
                "max": float(np.max(scores)),
                "p5": p5,
                "p95": p95,
            }
        return self

    def transform(self, strategy_id: str, scores: np.ndarray) -> np.ndarray:
        stats = self._stats.get(strategy_id)
        if stats is None:
            return scores
        if self.method == "minmax":
            lo, hi = stats["p5"], stats["p95"]
            if hi - lo < 1e-9:
                return np.zeros_like(scores)
            return np.clip((scores - lo) / (hi - lo), 0, 1)
        raise ValueError(f"Unknown calibration method: {self.method}")

    @property
    def is_fitted(self) -> bool:
        return len(self._stats) > 0

    def reset(self) -> None:
        self._stats.clear()
