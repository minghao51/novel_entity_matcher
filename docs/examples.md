# Examples

Related docs: [`index.md`](./index.md) | [`quickstart.md`](./quickstart.md) | [`experiments/index.md`](./experiments/index.md)

This repo keeps one active examples path: maintained examples for the supported APIs.

| Example | Focus |
|---|---|
| [`current/basic_matcher.py`](https://github.com/minghao51/novel_entity_matcher/blob/main/examples/current/basic_matcher.py) | Unified `Matcher`, async-first |
| [`current/trained_matcher.py`](https://github.com/minghao51/novel_entity_matcher/blob/main/examples/current/trained_matcher.py) | Supervised fitting with labeled examples |
| [`current/hierarchical_matching.py`](https://github.com/minghao51/novel_entity_matcher/blob/main/examples/current/hierarchical_matching.py) | Hierarchy-aware matching |
| [`novel_discovery_example.py`](https://github.com/minghao51/novel_entity_matcher/blob/main/examples/novel_discovery_example.py) | `NovelEntityMatcher` end-to-end discovery |
| [`pattern_strategy_example.py`](https://github.com/minghao51/novel_entity_matcher/blob/main/examples/pattern_strategy_example.py) | Pattern-based novelty scoring |
| [`oneclass_training_example.py`](https://github.com/minghao51/novel_entity_matcher/blob/main/examples/oneclass_training_example.py) | One-class novelty workflow |
| [`prototypical_training_example.py`](https://github.com/minghao51/novel_entity_matcher/blob/main/examples/prototypical_training_example.py) | Prototype-based novelty workflow |
| [`setfit_novelty_training_example.py`](https://github.com/minghao51/novel_entity_matcher/blob/main/examples/setfit_novelty_training_example.py) | SetFit novelty workflow |

Recommended order:

1. Start with [`current/basic_matcher.py`](https://github.com/minghao51/novel_entity_matcher/blob/main/examples/current/basic_matcher.py).
2. Use [`current/trained_matcher.py`](https://github.com/minghao51/novel_entity_matcher/blob/main/examples/current/trained_matcher.py) when you have labels.
3. Use [`novel_discovery_example.py`](https://github.com/minghao51/novel_entity_matcher/blob/main/examples/novel_discovery_example.py) for novelty detection.
4. Use `examples/raw/` only for lower-level experimentation.
