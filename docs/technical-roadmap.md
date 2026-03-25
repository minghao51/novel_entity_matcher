# Technical Roadmap

**Last Updated:** 2026-03-24  
**Status:** Active technical plan  
**Version Path:** 0.1.0 package today -> 1.0.0 architecture target

## Purpose

This document is the active technical source of truth for how `novel_entity_matcher` should evolve from its current matcher-first implementation into a modular discovery pipeline for entity matching, novelty detection, and human-reviewed class promotion.

It replaces the older milestone roadmap and the standalone modular-pipeline migration draft. Those documents are now archived so this file can stay tightly aligned with the current repository state.

## Current Repository Baseline

### Product and package reality today

- The published package metadata is still `0.1.0` in [`pyproject.toml`](./../pyproject.toml), even though older roadmap docs discuss `0.3.x`.
- The project already exposes two public surfaces:
  - `Matcher` as the primary general-purpose matching API
  - `NovelEntityMatcher` as the novelty-aware orchestration layer
- The README positions the library as a text-to-entity matcher with optional novelty detection, async support, benchmarking, and optional LLM-backed proposal features.
- The codebase already contains substantial novelty infrastructure; the gap is not "add novelty support" but "turn the current novelty stack into a clearer, pipeline-first architecture."

### Implemented architecture today

The current package layout is broader than the old roadmap summary implied:

- `core/`
  - unified `Matcher`
  - zero-shot, SetFit, BERT, and hybrid matching routes
  - async execution helpers
  - blocking, reranking, hierarchy, monitoring, normalization
- `novelty/`
  - modular detector core and strategy registry
  - strategy implementations for confidence, KNN distance, clustering, one-class, pattern, prototypical, self-knowledge, and SetFit-based novelty signals
  - clustering, proposal, schema, storage, config, evaluation, and utility subpackages
  - `NovelEntityMatcher` orchestration API
- `backends/`
  - static embeddings
  - sentence-transformer embeddings
  - reranker backend
  - LiteLLM integration
- `benchmarks/`
  - benchmark runner/registry/CLI scaffolding already exists in the package
- `ingestion/`
  - dataset-specific ingestion modules and CLI
- `tests/`
  - 31 test modules spanning core matchers, novelty behavior, backends, ingestion, packaging, and utilities

### Strengths already present

- Unified matcher API with sync and async flows
- Training-aware model resolution and static-embedding defaults
- A real novelty subsystem with pluggable strategies, evaluation helpers, schemas, and persistence
- ANN, clustering, and LLM proposal primitives already present
- Broad test coverage across the current public API surface
- Documentation for architecture, async usage, models, experiments, and troubleshooting

### Key technical gaps

- The public architecture is still matcher-first rather than pipeline-first.
- Novelty stages exist, but they are not expressed as a single explicit discovery pipeline contract.
- Current docs mix implemented capabilities with proposed future architecture.
- OOD handling is still strategy-driven rather than a dedicated stage contract with stable interfaces.
- Cluster-level extraction, schema-enforced proposal retries, and promotion workflows are not yet first-class end-to-end flows.
- HITL review and promotion remain persistence-oriented rather than operationally complete.

## Target Architecture

### North-star outcome

Ship a pipeline-first architecture that supports:

1. fast known-entity matching for routine traffic
2. structured routing of uncertain or out-of-distribution inputs
3. cluster-level discovery of candidate novel concepts
4. human-reviewed promotion of accepted concepts back into the reference system

### Target execution model

The long-term discovery flow should be:

1. Normalize and vectorize input text.
2. Run known-entity matching for standard retrieval/classification.
3. Route low-confidence or out-of-distribution items into a dedicated discovery path.
4. Cluster likely novel items into communities.
5. Extract concise statistical evidence for each cluster.
6. Run LLM proposal/judgment at the cluster level, not per sample.
7. Validate structured outputs and persist review-ready proposals.
8. Support review, approval, promotion, and retraining/index refresh.

### Architectural principles

- Pipeline-first orchestration should become the main internal design.
- Existing matcher and novelty functionality should be reused where it is already sound.
- Breaking API changes are acceptable when they simplify the architecture materially.
- Configuration should drive stage selection and optional capabilities.
- LLM usage should stay optional, bounded, and cost-aware.
- Promotion of new classes should be explicit and auditable.

## Current vs Target Gap Analysis

| Area | Current state | Target state | Gap |
|---|---|---|---|
| Matching API | `Matcher` is the main public interface | Matching becomes one subsystem within a broader pipeline | Medium |
| Novelty detection | Multi-strategy detector exists | Dedicated OOD/discovery stage contracts with clearer boundaries | Medium |
| Discovery orchestration | `NovelEntityMatcher` chains matcher, detector, proposer | Explicit pipeline orchestrator with named stages and stable stage I/O | High |
| Clustering | Scalable clustering exists | Clustering becomes a standard pipeline stage with interchangeable backends | Medium |
| Evidence extraction | No dedicated keyword/statistical extraction stage | Cluster summarization before LLM calls | High |
| LLM proposals | LLM proposer exists | Cluster-level judge/proposer with validation and retries | Medium |
| Schema enforcement | Pydantic models exist | Retry-aware schema enforcement and cleaner proposal contracts | Medium |
| HITL workflow | Persistence/export exists | Review queue, promotion workflow, index/model update path | High |
| Configuration | Current config/model registries exist | Stage-oriented pipeline configuration | Medium |
| Documentation | Multiple roadmap narratives, some stale | One active technical roadmap aligned to repo reality | Completed by this task |

## Roadmap Phases

### Phase 0: Documentation and Baseline Alignment

Goal: make the repository truth and the roadmap agree before major architecture work continues.

- Consolidate roadmap material into this document.
- Archive superseded roadmap drafts.
- Keep architecture docs honest about what is implemented versus planned.
- Treat the current public API as supported while the new internal architecture is shaped.
- Establish a habit that new technical design work updates this roadmap and the architecture doc together.

Exit criteria:

- one active roadmap document
- no contributor-facing docs still pointing to archived roadmap files as the current plan
- roadmap statements match current package structure, extras, and test reality

### Phase 1: Stabilize Current Matcher and Novelty Foundations

Goal: reduce ambiguity and technical drift before introducing a new orchestrator.

- Normalize internal contracts across `Matcher`, `NovelEntityMatcher`, novelty detector outputs, and match metadata.
- Tighten boundaries between:
  - known-entity matching
  - novelty scoring
  - proposal generation
  - persistence/reporting
- Audit the novelty strategy registry and mark which strategies are production-facing, experimental, or internal.
- Ensure async and sync paths produce equivalent metadata needed for downstream novelty/discovery stages.
- Expand tests around metadata contracts, top-k outputs, detector configuration, and persistence behavior.

Exit criteria:

- stable result objects and metadata shape for downstream discovery work
- documented boundaries between matching and novelty subsystems
- reduced doc drift around what strategies and backends are actually supported

### Phase 2: Introduce Pipeline Contracts Internally

Goal: establish the internal abstraction needed for staged discovery without breaking everything at once.

- Add a pipeline package or equivalent internal module defining:
  - stage context
  - stage result
  - stage lifecycle/init hooks
  - pipeline orchestrator contract
- Implement adapters around existing capabilities rather than rewriting them immediately:
  - embedding/vectorization adapter
  - matcher/routing adapter
  - OOD detection adapter
  - clustering adapter
  - proposal adapter
- Keep `Matcher` and `NovelEntityMatcher` usable, but start routing their internals through shared stage contracts where practical.
- Define the minimum configuration surface for stage selection and optional stage enablement.

Planned public/interface changes:

- Introduce a pipeline-first internal API before exposing a new top-level public class.
- Avoid exposing unstable stage internals until contracts settle.

Exit criteria:

- end-to-end internal discovery flow can execute through stage contracts
- existing public APIs continue to work or have explicit compatibility shims
- stage interfaces are documented well enough to support experimentation

### Phase 3: Upgrade Discovery-Specific Stages

Goal: improve the quality and efficiency of novel class discovery.

- Split discovery into explicit stages:
  - OOD filtering
  - community detection
  - cluster evidence extraction
  - proposal/judgment
  - schema enforcement
- Add stronger OOD methods where they materially outperform threshold-only logic.
  - candidate methods include Mahalanobis distance and local outlier approaches
- Keep current clustering support, but make clustering backends pluggable through a stage contract.
- Add a cluster summarization layer before LLM use.
  - statistical keywords
  - representative examples
  - token-budget-aware context packaging
- Refactor proposal generation toward cluster-level calls instead of sample-level prompting.

Planned public/interface changes:

- discovery report objects should expose stage outputs and diagnostics more explicitly
- proposal interfaces should accept cluster/evidence inputs rather than raw loose sample lists

Exit criteria:

- cluster-level discovery path works without per-sample LLM dependence
- structured proposal generation uses evidence extracted from clusters
- discovery cost and latency are meaningfully improved on benchmark scenarios

### Phase 4: Human-in-the-Loop Review and Promotion

Goal: make novel concept handling operational rather than just analytical.

- Introduce explicit review-state persistence for proposed classes.
- Model proposal lifecycle states such as:
  - pending review
  - approved
  - rejected
  - promoted
- Add promotion mechanics that can update the known reference index and trigger refresh/retrain workflows where required.
- Define safe promotion boundaries for:
  - embedding index updates
  - matcher retraining requirements
  - proposal provenance and audit metadata
- Provide a thin operational surface first.
  - CLI or programmatic review flows are enough initially
  - API/server/dashboard work can follow after the underlying lifecycle is stable

Planned public/interface changes:

- proposal storage moves from export-only behavior to lifecycle-aware review records
- promotion APIs become explicit, auditable operations rather than implied manual follow-up

Exit criteria:

- approved concepts can be promoted through a documented workflow
- promotion updates are traceable and testable
- rejected proposals do not silently re-enter the system as if they were unresolved

### Phase 5: Public Pipeline API and 1.0 Readiness

Goal: expose the final architecture cleanly and retire transitional seams.

- Introduce the long-term public orchestrator, likely a pipeline-first entry point such as `DiscoveryPipeline`.
- Decide the final relationship among:
  - `Matcher`
  - `NovelEntityMatcher`
  - the new pipeline API
- Consolidate configuration around stable pipeline concepts rather than ad hoc feature toggles.
- Remove or demote transitional wrappers once the final public story is clear.
- Finish packaging, examples, and docs for the final public API.

Public API expectations for 1.0:

- clear primary entry point for known matching
- clear primary entry point for discovery
- stable result models for match, novelty, cluster, and proposal outputs
- documented optional extras for novelty, LLM, visualization, and future review tooling

Exit criteria:

- one coherent public architecture
- upgrade path documented from current APIs
- benchmark and evaluation guidance updated for the final architecture
- docs no longer describe experimental internals as if they were the stable interface

## Planned Public API and Module Evolution

### Short term

- Keep `Matcher` as the recommended known-entity matching interface.
- Keep `NovelEntityMatcher` as the novelty-aware orchestration layer while internal pipeline contracts are introduced.
- Strengthen result/metadata contracts instead of adding many new top-level classes immediately.

### Medium term

- Introduce stage contracts and a reusable internal orchestrator.
- Refactor novelty and proposal modules around stage inputs/outputs rather than loose chaining.
- Add explicit review/promotion APIs and storage models.

### Long term

- Expose a pipeline-first API for discovery.
- Clarify whether `NovelEntityMatcher` remains as a friendly facade, becomes a compatibility shim, or is replaced entirely.
- Reorganize module boundaries so discovery stages, configuration, and promotion workflows are easier to reason about than the current mixed orchestration layout.

## Testing, Benchmarking, and Validation Work

Technical roadmap work is only complete when accompanied by verification.

### Testing priorities

- Preserve and extend unit coverage for current matcher and novelty behavior.
- Add contract tests for:
  - metadata emitted by matcher flows
  - novelty detector stage inputs/outputs
  - proposal schema validation
  - persistence and promotion lifecycle state transitions
- Add async/sync parity tests for any shared pipeline path.
- Keep LLM-dependent tests isolated and clearly marked.

### Benchmarking priorities

- Maintain baseline benchmarks for:
  - known-entity match quality
  - novelty detection quality
  - clustering quality for novel pools
  - end-to-end latency and cost for discovery runs
- Compare any new OOD strategy against the current confidence/KNN-driven baseline before adoption.
- Measure cluster-level proposal generation against sample-level prompting for both quality and cost.

### Documentation priorities

- Update [`architecture.md`](./architecture.md) whenever module boundaries materially change.
- Keep examples aligned with whichever public discovery API is current.
- Keep planning docs out of the user-guide path once they are archived.

## Sequencing Dependencies

- Documentation alignment must happen before more architectural divergence accumulates.
- Stable metadata/result contracts should precede major orchestrator refactors.
- Pipeline contracts should exist before large new stage implementations are introduced.
- Review/promotion workflows should build on stable proposal schemas and storage models.
- Final public API decisions should wait until the internal stage architecture has proven itself in tests and benchmarks.

## Risks and Mitigations

| Risk | Why it matters | Mitigation |
|---|---|---|
| Doc drift returns | The repo already contains roadmap statements that no longer match implementation details | Treat roadmap and architecture updates as part of feature work |
| Premature API redesign | A public pipeline API too early could freeze weak internal contracts | Build internal stage contracts first, then expose the final surface |
| Discovery complexity balloons | Novelty, clustering, LLM, and review flows can sprawl quickly | Keep stage boundaries explicit and verify each stage independently |
| LLM cost/latency creep | Proposal generation can become expensive at sample level | Prefer cluster-level evidence extraction and bounded prompting |
| Promotion workflow mistakes | Poorly governed promotion can corrupt the known entity index | Make approval state explicit and promotion auditable |
| Experimental method sprawl | Too many novelty strategies can blur the supported path | Label experimental vs supported paths and benchmark before promoting |

## Near-Term Implementation Priorities

These should guide the next few engineering cycles:

1. Keep docs, package metadata, and architecture notes aligned with the current repository.
2. Stabilize matcher-to-novelty metadata contracts used by downstream discovery logic.
3. Introduce internal pipeline contracts with minimal disruption to the current public API.
4. Upgrade discovery stages toward cluster-level evidence extraction and proposal generation.
5. Add review/promotion lifecycle support before attempting a polished dashboard or service layer.

## Definition of Done for This Roadmap

This roadmap remains the active source of truth until replaced. It should be updated whenever one of the following changes:

- the primary public API changes
- the stage architecture materially changes
- promotion/review workflows become more concrete
- package extras, supported backends, or test strategy change in a way that affects implementation planning

When a future roadmap replaces this one, it should supersede this document explicitly and move this file into the archive.
