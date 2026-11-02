# Experimental Protocol

## 1. Experimental Configurations (E1 – E4)

All four experimental configurations are executed on identical train/validation/test splits using consistent evaluation metrics:

| Experiment ID | Name | Retrieval Architecture | Verification Granularity | NLI Verification | Cross-Source Conflict Detection | LLM Judge Policy | Confidence Calibration |
|---|---|---|---|---|---|---|---|
| **E1** | `ORIGINAL_BASELINE` | Single pooled hybrid RAG | Collective whole-answer | Holistic context overlap / baseline faithfulness | Disabled | None (collective heuristic / Ragas-style) | Uncalibrated |
| **E2** | `CLAIM_LEVEL_NLI` | Single pooled hybrid RAG | Atomic claim-level | Pooled max-NLI across all chunks | Disabled | Disabled (0 calls) | Uncalibrated NLI score |
| **E3** | `HYBRID_VERIFICATION` | Single pooled hybrid RAG | Atomic claim-level | Pooled NLI + unconditional LLM Judge | Disabled | Always invoked (100% of claims) | Unweighted average |
| **E4** | `PROPOSED_ACVE` | 4 Independent Multi-RAG pipelines | Atomic claim-level | Source-wise NLI per RAG collection | Enabled (JSD + Polarity CSI matrix) | Adaptive gatekeeper (conflict/margin triggered) | Validation-tuned logistic + Temperature Scaling |

---

## 2. Datasets and Task Adaptation

1. **Controlled Multi-Source Conflict Benchmark (`ms_conflict_bench`)**:
   - Contains 60 multi-source verification instances (120+ atomic claims) spanning 6 controlled categories:
     - `consistent_support`: Multiple sources corroborate the claim.
     - `direct_contradiction`: Retrieved evidence directly refutes the claim (numerical, temporal, entity, or polarity contradiction).
     - `cross_source_conflict`: One source (e.g., outdated web blog) supports a claim while a higher-authority source (e.g., peer-reviewed paper or official telemetry PDF) contradicts it.
     - `partially_supporting`: Evidence supports one clause of a compound answer while contradicting or omitting the second clause.
     - `insufficient_evidence`: Retrieved passages are topically related but lack the specific fact claimed.
     - `semantic_trap`: Evidence has $>85\%$ lexical overlap with the claim but opposite factual meaning (e.g., negation or counterfactual).
2. **RAGTruth-Compatible Benchmark (`ragtruth_eval`)**:
   - QA and summarization instances formatted with span- and claim-level hallucination annotations (`supported` vs. `hallucinated` / `subtle_contradiction`).
3. **HaluEval-Compatible Benchmark (`halueval_qa`)**:
   - Knowledge-grounded QA instances containing paired faithful and hallucinated responses given retrieved passages.
4. **RefChecker / Multi-Hop QA Benchmark (`multihop_conflict_qa`)**:
   - Multi-hop reasoning claims adapted from HotpotQA / 2WikiMultiHopQA structures requiring evidence synthesis across two distinct documents.

> **Dataset Integrity Disclosure**: Public benchmarks such as HaluEval and RAGTruth annotate hallucinations against a single provided reference context and do **not** natively include 4-way cross-source conflict labels. Only `ms_conflict_bench` and `multihop_conflict_qa` contain explicit multi-source provenance and inter-source conflict ground truth.

---

## 3. Data Split & Anti-Leakage Protocol

- Each dataset is partitioned deterministically (seed `42`) into:
  - **Validation Split (40%)**: Used exclusively to fit the logistic aggregation weights $\mathbf{w}$ and temperature scaling parameter $T$.
  - **Test Split (60%)**: Held out strictly for reporting all final classification, calibration, efficiency, and evidence metrics.
- Test labels are never accessed during weight fitting, threshold selection, or prediction generation.
