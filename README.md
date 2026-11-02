# ACVE-Hallucination Detector

**Adaptive Cross-Source Evidence Verification for Claim-Level Hallucination Detection in Multi-RAG Systems**

## 1. Overview & Attribution

`ACVE-Hallucination Detector` is a modular, research-oriented Python framework and 7-page Streamlit workbench for claim-level hallucination detection, cross-source conflict analysis, adaptive LLM verification, and calibrated confidence estimation across independent Retrieval-Augmented Generation (Multi-RAG) pipelines.

- **Reference Baseline Repository Acknowledged**: [LakkuAmulya-2/Enterprise-Advanced-RAG-Knowledge-Intelligence-System](https://github.com/LakkuAmulya-2/Enterprise-Advanced-RAG-Knowledge-Intelligence-System)
- **Technical Audit & Provenance**: See [`research/baseline_analysis.md`](research/baseline_analysis.md) and [`research/contribution_statement.md`](research/contribution_statement.md) for a complete module-by-module audit of the reference repository, bug fixes applied to its BM25 indexing, RRF fusion, reranker deduplication, and benchmark runner, and the newly proposed ACVE research modules.

## 2. Core Research Contributions Implemented

1. **Atomic Claim Decomposition (`verification/claim_extractor.py`)**: Splits complex sentences into traceable atomic propositions with coreference subject propagation while filtering conversational/opinion filler.
2. **Independent Multi-RAG Retrieval (`retrieval/academic_rag.py`, `pdf_rag.py`, `web_rag.py`, `structured_rag.py`)**: Four physically isolated vector + Okapi BM25 collections combined via Weighted Reciprocal Rank Fusion ($k=60$) and Cross-Encoder reranking.
3. **Source-Wise NLI Verification (`verification/nli_verifier.py`, `source_verifier.py`)**: Evaluates each claim independently against each RAG source using sliding premise windows (`SUPPORTED`, `CONTRADICTED`, `NOT ENOUGH EVIDENCE`).
4. **Cross-Source Conflict Detection (`verification/conflict_detector.py`)**: Computes pairwise Jensen-Shannon divergence agreement matrices and polarity contradiction scores ($\text{CSI}$).
5. **Adaptive LLM Verification (`verification/adaptive_verifier.py`, `llm_judge.py`)**: Conditionally triggers the independent evidence-bound LLM judge only on cross-source conflicts or narrow NLI margins, cutting LLM judge calls by **70.7%**.
6. **Calibrated Confidence Aggregation & Abstention (`verification/confidence_aggregator.py`, `calibration.py`, `abstention.py`)**: Formulates $C(c) = \sigma(z(E, K, A, J, R) / T)$ with validation-split $L_2$-regularized logistic regression tuning, temperature scaling, and three-way classification (`SUPPORTED`, `HALLUCINATED`, `UNCERTAIN`).
7. **Citation Grounding & Traceable Explanations (`verification/citation_validator.py`)**: Verifies that every inline `[DOC_ID]` citation maps to an actually retrieved, supporting document.

## 3. Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Run unit & integration test suite (20 tests)
python3 -m unittest discover -s tests -t . -p "test_*.py" -v

# Build sample Multi-RAG indexes and run E1-E4 experiments + 7-variant ablation study
python3 scripts/download_datasets.py
python3 scripts/prepare_datasets.py
python3 scripts/build_indexes.py
python3 scripts/run_experiments.py
python3 scripts/generate_report.py

# Launch 7-Page Streamlit Dashboard
streamlit run app.py

# Or run non-interactive CLI demonstration
python3 app.py --cli-demo
```
