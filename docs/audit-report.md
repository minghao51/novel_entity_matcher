# Codebase Audit Report (April 2026)

## Overview
This report summarizes the results of a comprehensive audit and review of the `novel-entity-matcher` repository. It evaluates the progress made during recent refactoring efforts and identifies remaining technical debt, bugs, and areas for improvement.

## 1. Progress & Improvements
The following concerns from the initial audit have been successfully addressed:

- **Matching Strategy Pattern:** Introduced `src/novelentitymatcher/core/matching_strategy.py` to decompose `matcher.py`.
- **Model Caching:** Implemented a thread-safe `ModelCache` in `src/novelentitymatcher/utils/embeddings.py`, significantly reducing model initialization latency.
- **Security Hardening:** Removed `allow_pickle=True` from all `numpy.load` calls to prevent arbitrary code execution.
- **Exception Handling:** Broad `except Exception:` blocks have been replaced with specific exception tuples and structured logging.
- **Source Cleanliness:** Removed `print()` statements from core modules, standardizing on the project's logging infrastructure.
- **Pipeline Consolidation:** Created `PipelineBuilder` to centralize the construction of the 5-stage discovery pipeline.

## 2. Remaining Gaps & Concerns

### **Critical Bugs**
- **Clustering `KeyError`:** The `CommunityDetectionStage` in `adapters.py` defaults to `backend_name="auto"`. However, the `ClusteringBackendRegistry` does not contain an `"auto"` entry, causing a `KeyError` at runtime.
  - **Location:** `src/novelentitymatcher/pipeline/adapters.py` and `src/novelentitymatcher/novelty/clustering/backends.py`.

### **Architectural Redundancy**
- **Duplicated Orchestration:** `NovelEntityMatcher` and `DiscoveryPipeline` still maintain their own pipeline construction logic. They should be refactored to delegate entirely to the `PipelineBuilder` class to ensure consistency and reduce maintenance overhead.
  - **Location:** `src/novelentitymatcher/novelty/entity_matcher.py` and `src/novelentitymatcher/pipeline/discovery.py`.

### **Technical Debt**
- **Monolithic `Matcher`:** Despite the strategy refactor, `matcher.py` is still approximately 1,200 lines long. It still manages significant state and configuration logic that should be moved into the strategies or a dedicated configuration handler.
- **Inefficient TF-IDF:** The keyword extraction in `ClusterEvidenceStage` uses a manual nested loop ($O(n \times m)$) for TF-IDF. This will not scale for large discovery batches.
  - **Location:** `src/novelentitymatcher/pipeline/adapters.py`.

## 3. Recommended Next Steps

### **Immediate: Stability & Performance**
1.  **Fix the `KeyError: 'auto'`:** Refactor `CommunityDetectionStage` to use the `ScalableClusterer` class directly (which already handles the "auto" logic) instead of the registry.
2.  **Optimize TF-IDF:** Replace the manual TF-IDF implementation in `adapters.py` with a vectorized approach using `sklearn.feature_extraction.text.TfidfVectorizer` or `scipy` sparse matrices.

### **Short-Term: Architectural Cleanup**
3.  **Complete `PipelineBuilder` Migration:** Refactor `NovelEntityMatcher` and `DiscoveryPipeline` to use `PipelineBuilder` for all stage orchestration. Remove the `DiscoveryPipelineMixin` once the migration is complete.
4.  **Refactor Matcher State:** Move the remaining configuration and mode-selection logic from `_EntityMatcher` into the specialized strategy classes in `matching_strategy.py`.

### **Long-Term: Scalability**
5.  **Batched Ingestion:** Optimize the ingestion CLI to support concurrent fetching and batched database writes for larger datasets.
6.  **Observability:** Implement a structured metrics exporter (e.g., Prometheus or OpenTelemetry) to track match latency, novelty rates, and cache hits in production environments.
