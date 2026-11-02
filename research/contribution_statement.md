# Research Contribution Statement

## 1. Distinction Between Inherited Baseline and Proposed Contributions

In accordance with academic software engineering and reproducibility standards, this document explicitly separates components inherited or adapted from the reference repository ([LakkuAmulya-2/Enterprise-Advanced-RAG-Knowledge-Intelligence-System](https://github.com/LakkuAmulya-2/Enterprise-Advanced-RAG-Knowledge-Intelligence-System)) from established NLP techniques and newly proposed system contributions in **ACVE-Hallucination Detector**.

### 1.1 Inherited & Corrected from Reference Repository
- **Okapi BM25 Formulation**: Adapted from `src/retrieval/sparse_retriever.py`, with bug fixes for punctuation stripping, incremental document addition/deletion, and empty-index zero-division protection.
- **Weighted Reciprocal Rank Fusion (RRF)**: Adapted from `src/retrieval/hybrid_fusion.py` ($k=60$), extended to retain per-retriever diagnostic scores.
- **Cross-Encoder Reranking Pattern**: Adapted from `src/reranking/cross_encoder.py`, replacing the buggy 200-character prefix deduplication with shingle Jaccard similarity.

### 1.2 Established Literature Techniques Integrated
- **Atomic Fact/Claim Decomposition**: Inspired by FactScore (Min et al., 2023) and RefChecker (Hu et al., 2024), implemented via deterministic syntax/clause splitting and LLM-assisted extraction.
- **NLI-based Entailment Scoring**: Following Honovich et al. (2022, TRUE) and Laban et al. (2022, SummaC), using DeBERTa-v3 NLI / calibrated structural entailment over chunked premises.
- **Post-hoc Temperature Scaling Calibration**: Following Guo et al. (2017) for Expected Calibration Error (ECE) reduction.

### 1.3 Proposed Research Extensions in ACVE
1. **Source-Isolated Multi-RAG Verification Topology**: Rather than pooling chunks from a single vector store into an undifferentiated context window, ACVE maintains four distinct retrieval pipelines (`academic`, `pdf`, `web`, `structured`) and preserves source boundaries through the entire verification lifecycle.
2. **Cross-Source Conflict Matrix & Polarity-Divergence Scoring**: We introduce the Cross-Source Conflict Severity Index ($\text{CSI}$) combining pairwise Jensen-Shannon divergence with entailment-contradiction cross-products $\Gamma_{a,b} = E_a K_b + K_a E_b$, explicitly separating source reliability ($R$), retrieval relevance ($s_{\text{retrieval}}$), evidence strength ($E, K$), and inter-source agreement ($A$).
3. **Conflict- and Margin-Gated Adaptive LLM Verification**: Instead of invoking an LLM judge on 100% of claims (as in pure LLM-as-a-judge pipelines), ACVE triggers the LLM judge conditionally when $\text{CSI}(c_i) \ge \theta_{\text{conflict}}$ or the NLI entailment-contradiction margin is ambiguous, reducing LLM verification calls while targeting high-risk disagreements.
4. **Calibrated Three-Way Decision with Subtyped Abstention**: Distinguishes between claims that are directly contradicted by retrieved evidence (`HALLUCINATED: DIRECTLY_CONTRADICTED`), fabricated without support (`HALLUCINATED: UNSUPPORTED_FABRICATION`), and claims requiring abstention due to insufficient or irreconcilably conflicting evidence (`UNCERTAIN`).
