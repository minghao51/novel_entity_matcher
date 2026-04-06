# Novel Class Detection

Related docs: [`quickstart.md`](./quickstart.md) | [`architecture.md`](./architecture.md) | [`technical-roadmap.md`](./technical-roadmap.md)

`NovelEntityMatcher` is the supported orchestration API for novelty-aware matching and class discovery. This page is the current source of truth for using the novelty stack that already exists in the repo today.

## When to Use It

Use `Matcher` by itself when every query should map to an existing entity. Add `NovelEntityMatcher` when some queries may represent:

- a genuinely new class
- an out-of-distribution input that should be reviewed instead of force-matched
- a cluster of related unmatched samples that should be proposed as a new entity

## Main Flow

```python
from novelentitymatcher import Matcher, NovelEntityMatcher
from novelentitymatcher.novelty import DetectionConfig
from novelentitymatcher.novelty.config.strategies import ConfidenceConfig, KNNConfig

entities = [
    {"id": "physics", "name": "Physics"},
    {"id": "cs", "name": "Computer Science"},
]

matcher = Matcher(entities=entities, model="minilm", threshold=0.6)
matcher.fit(
    texts=["quantum mechanics", "neural networks"],
    labels=["physics", "cs"],
)

novel_matcher = NovelEntityMatcher(
    matcher=matcher,
    detection_config=DetectionConfig(
        strategies=["confidence", "knn_distance"],
        confidence=ConfidenceConfig(threshold=0.65),
        knn_distance=KNNConfig(distance_threshold=0.45),
    ),
    auto_save=False,
)
```

```python
report = await novel_matcher.discover_novel_classes(
    queries=["quantum biology", "new interdisciplinary topic"],
    existing_classes=["physics", "cs"],
    run_llm_proposal=False,
)
```

## What Exists Today

The current novelty stack is already broader than a simple threshold check:

- `NovelEntityMatcher` orchestrates match, detect, cluster, and optional proposal steps
- `NoveltyDetector` combines multiple signals instead of relying on one threshold
- distance-based novelty can use ANN-backed search structures
- proposal and review helpers can persist artifacts for later inspection

## Detection Strategies

The exact strategy set may continue evolving, but the active system centers on these kinds of signals:

- confidence and uncertainty signals from the base matcher
- nearest-neighbor distance and support checks against known examples
- clustering-based grouping for batches of suspicious samples
- optional proposal generation for naming or summarizing candidate new classes

The practical recommendation is to start with conservative confidence and k-NN thresholds, then add clustering or proposal steps once you have a review workflow.

## Lower-Level Components

- `novelentitymatcher.novelty.core.detector.NoveltyDetector`: modular detector used by `NovelEntityMatcher`
- `novelentitymatcher.novelty.proposal.llm.LLMClassProposer`: LLM-backed naming and summarization
- `novelentitymatcher.novelty.storage.ANNIndex`: ANN search index used by distance-based strategies
- `novelentitymatcher.novelty.storage.save_proposals` / `load_proposals`: persistence helpers for discovery reports

## Reports

Discovery returns `NovelClassDiscoveryReport`, which typically contains:

- `novel_sample_report.novel_samples`: flagged samples with confidence, novelty score, signals, and per-sample metrics
- `class_proposals`: optional generated class names and justifications
- `metadata`: counts and output paths for saved artifacts

## Example Paths

Start with the maintained end-to-end example:

```bash
uv run python examples/novel_discovery_example.py
```

Other novelty-oriented examples in `examples/` cover pattern-based, one-class, prototypical, and SetFit-backed workflows. Use them as implementation examples, not as separate sources of truth for the public docs.

## Notes

- Use [`technical-roadmap.md`](./technical-roadmap.md) for future architectural direction.
- Older research-heavy novelty proposals have been moved into the archive so this page can stay focused on the current workflow.
