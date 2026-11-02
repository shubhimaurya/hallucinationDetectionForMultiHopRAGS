# Module-by-Module Implementation Guide

## 1. Configuration Subsystem (`config/`)
- `config/settings.py`: Defines `Settings`, managing paths, LLM provider configurations (`local_mock`, `openai`, `ollama`), embedding/reranker/NLI model identifiers, chunk sizes, and API key masking (`_mask_secret`) so credentials never leak to logs.
- `config/model_config.py`: Defines `RAG_SOURCE_CONFIGS` (reliability priors and metadata for `academic`, `pdf`, `web`, and `structured` RAG collections) and `VerificationConfig` (thresholds, weights $\beta_0, w_E, w_K, w_A, w_J, w_R, w_{\text{CSI}}$, temperature $T$, and ablation switches).

## 2. Document Ingestion Subsystem (`ingestion/`)
- `ingestion/document_loader.py`: Parses `.pdf`, `.txt`, `.docx`, `.csv`, and `.md` files. Enforces file-size and extension validation, computes SHA-256 content hashes, preserves page/section provenance, and neutralizes prompt-injection attempts via `sanitize_untrusted_text()`.
- `ingestion/chunking.py`: Implements `OverlappingChunker` and `split_sentences_safe()`, protecting scientific abbreviations (`Dr.`, `Fig.`, `et al.`, `U.S.`, decimals) before chunking with configurable overlap.
- `ingestion/embedding.py`: Implements `EmbeddingGenerator`, loading `SentenceTransformer` when available and providing a deterministic 384-dimensional Johnson-Lindenstrauss signed random projection over unigrams, canonical synonyms, bigrams, and character 4-grams for offline reproducibility.
- `ingestion/indexing.py`: Implements `SourceCollectionIndex`, maintaining physically isolated vector arrays (`embeddings.npy`) and incremental Okapi BM25 inverted indexes per RAG source. Fixes the reference repository's BM25 overwrite bug and punctuation tokenization bug.
- `ingestion/document_processor.py`: Coordinates end-to-end ingestion, duplicate detection, document deletion, and re-indexing across all four collections.

## 3. Multi-RAG Retrieval Subsystem (`retrieval/`)
- `retrieval/academic_rag.py` (`AcademicRAGPipeline`), `retrieval/pdf_rag.py` (`PDFRAGPipeline`), `retrieval/web_rag.py` (`WebRAGPipeline`), `retrieval/structured_rag.py` (`StructuredRAGPipeline`): Four independent retrieval pipelines bound to isolated `SourceCollectionIndex` stores with distinct reliability priors.
- `retrieval/fusion.py`: Implements `reciprocal_rank_fusion()` ($k=60$), retaining both `dense_score` and `sparse_score` without overwriting metadata.
- `retrieval/reranker.py`: Implements `CrossEncoderReranker` (`cross-encoder/ms-marco-MiniLM-L-6-v2` + lexical-semantic interaction scorer) and fixes the reference repository's 200-character prefix deduplication bug using 3-word shingle Jaccard similarity.
- `retrieval/hybrid_retriever.py`: Implements `MultiRAGHybridRetriever`, allowing individual pipelines to be enabled/disabled dynamically.

## 4. Answer Generation Subsystem (`generation/`)
- `generation/prompt_templates.py`: Wraps retrieved evidence inside `<untrusted_evidence>` delimiters with explicit instructions treating retrieved passages as passive data rather than executable commands.
- `generation/answer_generator.py`: Implements `AnswerGenerator` with support for OpenAI-compatible endpoints, local Ollama servers, and a deterministic extractive synthesizer that attaches `[DOC_ID]` citations and abstains when evidence relevance falls below threshold.

## 5. Verification Subsystem (`verification/`)
- `verification/claim_extractor.py`: Implements `AtomicClaimExtractor` (Contribution 1), splitting compound sentences, propagating nominal subjects across pronouns, and filtering conversational/opinion statements.
- `verification/nli_verifier.py`: Implements `NLIVerifier` (Contribution 3), chunking long premises into sliding windows and computing tri-class probabilities $(p_{\text{ent}}, p_{\text{con}}, p_{\text{neu}})$ with explicit negation, antonym, entity, and numeric verification.
- `verification/source_verifier.py`: Implements `SourceWiseVerifier`, evaluating each atomic claim separately against each active RAG source before aggregation.
- `verification/conflict_detector.py`: Implements `CrossSourceConflictDetector` (Contribution 4), computing pairwise Jensen-Shannon divergence agreement matrices, polarity clash $\Gamma_{a,b}$, and the Conflict Severity Index ($\text{CSI}$).
- `verification/llm_judge.py` & `verification/adaptive_verifier.py`: Implement `IndependentLLMJudge` and `AdaptiveLLMVerifier` (Contribution 5), triggering the judge only on cross-source conflicts or narrow NLI margins.
- `verification/confidence_aggregator.py`, `verification/calibration.py`, & `verification/abstention.py`: Implement $C(c) = \sigma(z(E, K, A, J, R) / T)$ with validation-split $L_2$-regularized logistic regression tuning, post-hoc temperature scaling, and three-way abstention (`SUPPORTED`, `HALLUCINATED`, `UNCERTAIN`) (Contribution 6).
- `verification/citation_validator.py`: Implements `CitationValidator` and `ACVEPipelineVerifier` (Contribution 7), auditing inline citations against actually retrieved document IDs.

## 6. Evaluation & Dashboard Subsystems (`evaluation/`, `dashboard/`, `app.py`)
- `evaluation/datasets.py`, `metrics.py`, `baseline.py`, `experiments.py`, `ablation.py`, `comparison.py`, `visualization.py`: Complete benchmark suite for E1–E4 and 7 ablation variants.
- `dashboard/` & `app.py`: 7-page Streamlit interface plus `--cli-demo` non-interactive mode.
