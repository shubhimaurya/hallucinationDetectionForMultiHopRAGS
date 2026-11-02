# Research Methodology: Adaptive Cross-Source Evidence Verification (ACVE)

## 1. Problem Formulation

Let $Q$ denote a user query submitted to a Multi-RAG system comprising $M = 4$ independent knowledge repositories:
$$\mathcal{S} = \{S_{\text{academic}}, S_{\text{pdf}}, S_{\text{web}}, S_{\text{structured}}\}$$

For a query $Q$, each active retrieval pipeline $S_m \in \mathcal{S}$ retrieves a ranked set of evidence chunks:
$$\mathcal{D}_m(Q) = \{d_{m,1}, d_{m,2}, \dots, d_{m,k_m}\}$$
where each chunk $d_{m,j}$ retains structured provenance metadata:
$$\text{meta}(d_{m,j}) = (\text{source\_id}, \text{doc\_id}, \text{title}, \text{doc\_type}, s_{\text{retrieval}}, \text{page}, \text{reference}, r_m)$$
with $r_m \in (0, 1]$ denoting the prior source reliability metadata of source collection $S_m$ (configurable and ablatable).

Given the pooled reranked context $\mathcal{D}(Q) = \bigcup_{m=1}^M \mathcal{D}_m(Q)$, a language model generates a candidate response $Y$. Traditional RAG evaluation estimates a single collective faithfulness score $\Phi(Y, \mathcal{D}(Q)) \in [0, 1]$. As demonstrated in our baseline analysis, collective scoring fails when:
1. $Y$ contains multiple atomic propositions $C = \{c_1, \dots, c_L\}$ where a subset $C_{\text{false}} \subset C$ is hallucinated while $C \setminus C_{\text{false}}$ is supported.
2. Source $S_a$ entails $c_i$ while source $S_b$ contradicts $c_i$ (inter-source conflict).
3. High lexical/semantic overlap masks factual negation, numerical distortion, or entity swap.

Our objective is to learn a calibrated claim-level verification function:
$$\Psi(c_i, \{\mathcal{D}_m(Q)\}_{m=1}^M) \rightarrow (\hat{y}_i, \hat{p}_i, \mathcal{E}_i)$$
where $\hat{y}_i \in \{\texttt{SUPPORTED}, \texttt{HALLUCINATED}, \texttt{UNCERTAIN}\}$, $\hat{p}_i \in [0, 1]$ is the calibrated truthfulness probability, and $\mathcal{E}_i$ is a structured, citation-grounded explanation isolating supporting and contradictory evidence by source.

---

## 2. Algorithm 1: Atomic Claim Decomposition (`verification/claim_extractor.py`)

### 2.1 Design Principles
An atomic claim $c_i$ is an elementary declarative proposition stating a single verifiable relationship between entities, quantities, or events.

### 2.2 Decomposition Steps
1. **Abbreviation-Safe Sentence Segmentation**: Protect periods in common abbreviations (`Dr.`, `Prof.`, `Fig.`, `et al.`, `U.S.`, `v.`, decimal numbers `3.14`) before splitting the generated response $Y$ into sentences $s_1, \dots, s_N$ with exact character offsets $[\text{start}_n, \text{end}_n]$.
2. **Non-Factual Discourse Filtering**: Discard purely conversational greetings, meta-disclaimers (e.g., *"I hope this helps"*, *"Feel free to ask"*), and subjective value judgments lacking verifiable propositions.
3. **Coordinated Clause & Conjunction Splitting**:
   - Split compound sentences joined by coordinating conjunctions (`", and "`, `", while "`, `", whereas "`, `"; "`, `", but "`) when both clauses contain an explicit subject-predicate structure.
   - Apply **Coreference / Subject Propagation**: When the second clause begins with a third-person pronoun (`it`, `they`, `he`, `she`, `this mission`, `the system`) or an elided subject following a conjunction, resolve and substitute the canonical nominal subject from the preceding clause so that each extracted claim is self-contained (e.g., *"India launched Chandrayaan-3 in 2023, and it successfully landed near the lunar south pole"* $\rightarrow$ Claim 1: *"India launched Chandrayaan-3 in 2023."* and Claim 2: *"Chandrayaan-3 successfully landed near the lunar south pole."*).
4. **Traceability Schema**: Each claim is assigned a deterministic identifier `CLM-001`, `CLM-002`, $\dots$, recording `claim_id`, `claim_text`, `original_sentence`, `char_start`, `char_end`, and `cited_doc_ids`.

---

## 3. Algorithm 2: Source-Wise NLI Verification (`verification/nli_verifier.py`, `verification/source_verifier.py`)

Unlike cosine similarity between dense embeddings—which assigns high similarity to *"Chandrayaan-3 landed in 2023"* and *"Chandrayaan-3 failed to land in 2023"*—Natural Language Inference (NLI) models directional semantic entailment.

For each atomic claim $c_i$ and each active RAG source $S_m$:
1. **Evidence Chunking**: Long retrieved chunks $d_{m,j} \in \mathcal{D}_m(Q)$ exceeding the NLI window (256 tokens) are segmented into overlapping sliding windows $w_{m,j,1}, \dots, w_{m,j,V}$.
2. **Tri-Class Probability Estimation**: For each premise window $w$ and hypothesis $c_i$, the NLI model computes:
   $$\mathbf{p}(c_i \mid w) = (p_{\text{ent}}(c_i \mid w),\; p_{\text{con}}(c_i \mid w),\; p_{\text{neu}}(c_i \mid w)), \quad \sum_{z} p_z = 1$$
   - **Model Backend**: Supports HuggingFace DeBERTa-v3 NLI (`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` / `cross-encoder/nli-deberta-v3-small`) when available in the runtime environment, and includes a calibrated **Linguistic-Structural NLI Engine** for offline/CPU-constrained execution that explicitly evaluates entity alignment, numerical/temporal exactness, negation polarity inversion, antonym/quantifier contradiction, and predicate entailment (strictly forbidding raw cosine similarity as entailment).
3. **Source-Level Evidence Pooling**: Within source $S_m$, let:
   $$E_m(c_i) = \max_{w \in \mathcal{D}_m} p_{\text{ent}}(c_i \mid w), \quad K_m(c_i) = \max_{w \in \mathcal{D}_m} p_{\text{con}}(c_i \mid w)$$
   Source $S_m$ is classified as:
   - `CONTRADICTED` if $K_m(c_i) \ge \tau_{\text{con}}$ and $K_m(c_i) > E_m(c_i) + \delta_{\text{margin}}$
   - `SUPPORTED` if $E_m(c_i) \ge \tau_{\text{ent}}$ and $E_m(c_i) > K_m(c_i)$
   - `NOT ENOUGH EVIDENCE` otherwise.

---

## 4. Algorithm 3: Cross-Source Conflict Detection (`verification/conflict_detector.py`)

Given the source-wise verification vector $\mathbf{V}(c_i) = \{(S_m, \text{status}_m, E_m(c_i), K_m(c_i), N_m(c_i))\}_{m \in \mathcal{S}_{\text{active}}}$:

1. **Pairwise Agreement & Contradiction Matrices**:
   For every pair of active sources $(S_a, S_b)$:
   - **Agreement Score**:
     $$A_{a,b}(c_i) = 1 - \text{JSD}\big(\mathbf{p}_a(c_i) \,\|\, \mathbf{p}_b(c_i)\big)$$
     where $\text{JSD}$ is the normalized Jensen-Shannon divergence in $[0, 1]$ between the source-level probability distributions $\mathbf{p}_a = (E_a, K_a, N_a)$ and $\mathbf{p}_b = (E_b, K_b, N_b)$.
   - **Direct Polarity Conflict Score**:
     $$\Gamma_{a,b}(c_i) = E_a(c_i) \cdot K_b(c_i) + K_a(c_i) \cdot E_b(c_i)$$
2. **Conflict Severity Index**:
   $$\text{CSI}(c_i) = \max_{a \neq b} \Gamma_{a,b}(c_i) \in [0, 1]$$
3. **Cross-Source Agreement Score**:
   Let $\mathcal{S}_{\text{ev}}(c_i) = \{S_m : E_m(c_i) \ge 0.35 \text{ or } K_m(c_i) \ge 0.35\}$ be the set of sources bearing substantive evidence. If $|\mathcal{S}_{\text{ev}}| \ge 2$:
   $$A(c_i) = \frac{2}{|\mathcal{S}_{\text{ev}}|(|\mathcal{S}_{\text{ev}}| - 1)} \sum_{a < b \in \mathcal{S}_{\text{ev}}} A_{a,b}(c_i) \cdot (1 - \Gamma_{a,b}(c_i))$$
   If $|\mathcal{S}_{\text{ev}}| = 1$, $A(c_i) = 0.5$ (single-source uncorroborated). If $|\mathcal{S}_{\text{ev}}| = 0$, $A(c_i) = 0.0$.
4. **Conflict Taxonomy**:
   - `CONSENSUS_SUPPORT`: $\ge 2$ sources `SUPPORTED`, $0$ `CONTRADICTED`.
   - `SINGLE_SOURCE_SUPPORT`: $1$ source `SUPPORTED`, others `NOT ENOUGH EVIDENCE`.
   - `CROSS_SOURCE_CONTRADICTION`: At least one source `SUPPORTED` and at least one source `CONTRADICTED` (or $\text{CSI}(c_i) \ge 0.30$).
   - `CONSENSUS_CONTRADICTION`: $\ge 1$ source `CONTRADICTED`, $0$ `SUPPORTED`.
   - `INSUFFICIENT_EVIDENCE`: All active sources return `NOT ENOUGH EVIDENCE`.

Crucially, **majority voting is not used to erase conflicts**: if 2 blogs say X and 1 academic paper says $\neg$X, the conflict is flagged as `CROSS_SOURCE_CONTRADICTION` and routed to adaptive verification.

---

## 5. Algorithm 4: Adaptive LLM Verification (`verification/adaptive_verifier.py`, `verification/llm_judge.py`)

Invoking an LLM judge on every claim is computationally expensive. ACVE uses an uncertainty-and-conflict gatekeeper $\mathcal{G}(c_i) \in \{0, 1\}$:

$$\mathcal{G}(c_i) = \mathbb{I}\Big[ \underbrace{\text{CSI}(c_i) \ge \theta_{\text{conflict}}}_{\text{Cross-source disagreement}} \;\lor\; \underbrace{|\bar{E}(c_i) - \bar{K}(c_i)| < \theta_{\text{margin}}}_{\text{Ambiguous NLI margin}} \;\lor\; \underbrace{\max_m E_m(c_i) \in [\theta_{\text{low}}, \theta_{\text{high}}]}_{\text{Moderate uncorroborated evidence}} \Big]$$

- **Bypass Cases ($\mathcal{G}(c_i) = 0$)**:
  1. **High-Confidence Agreement**: $\max_m E_m(c_i) \ge \theta_{\text{high}}$ (e.g., $0.80$), $\text{CSI}(c_i) < \theta_{\text{conflict}}$, and $0$ contradicting sources $\rightarrow$ Directly verified without an LLM call.
  2. **Total Evidence Absence**: $\max_m E_m(c_i) < \theta_{\text{low}}$ (e.g., $0.25$) and $\max_m K_m(c_i) < \theta_{\text{low}}$ across all sources $\rightarrow$ Directly abstained (`UNCERTAIN` due to insufficient evidence) without wasting an LLM call.
- **Invocation Case ($\mathcal{G}(c_i) = 1$)**:
  The independent LLM judge receives the atomic claim $c_i$ and the separated source-wise evidence chunks (never relying on unretrieved external memory) and outputs structured JSON: `claim_id`, `decision`, `supporting_evidence`, `contradictory_evidence`, `explanation`, `confidence`, and `uncertainty_reason`.

---

## 6. Mathematical Formulation of Confidence Aggregation & Calibration (`verification/confidence_aggregator.py`, `verification/calibration.py`)

### 6.1 Feature Variables
For each claim $c_i$, we define five bounded, interpretable variables in $[0, 1]$:
1. **NLI Evidence Strength ($E$)**: Reliability-weighted maximum entailment across active sources:
   $$E(c_i) = \max_{m \in \mathcal{S}_{\text{active}}} p_{\text{ent}, m}(c_i)$$
2. **Contradiction Strength ($K$)**: Maximum contradiction probability across active sources:
   $$K(c_i) = \max_{m \in \mathcal{S}_{\text{active}}} p_{\text{con}, m}(c_i)$$
3. **Cross-Source Agreement ($A$)**: Jensen-Shannon and polarity-adjusted agreement $A(c_i) \in [0, 1]$ from Section 4.
4. **Independent Judge Assessment ($J$)**: Signed support score from the LLM judge when invoked ($J(c_i) \in [0, 1]$, where $1$ is full support, $0$ is contradiction, $0.5$ is neutral/not invoked). When the judge is bypassed due to high-confidence NLI consensus, $J(c_i)$ defaults to the NLI consensus posteriors so that no artificial penalty is introduced.
5. **Source Reliability Metadata ($R$)**: Reliability prior of the strongest supporting source vs. strongest contradicting source:
   $$R(c_i) = \frac{\sum_{m} r_m \cdot p_{\text{ent}, m}(c_i)}{\sum_{m} r_m \cdot (p_{\text{ent}, m}(c_i) + p_{\text{con}, m}(c_i)) + \epsilon}$$

### 6.2 Aggregation Function
We formulate the uncalibrated veracity logit $z(c_i)$ as a regularized linear combination of the five evidence features plus a conflict interaction penalty:
$$z(c_i) = \beta_0 + w_E E(c_i) - w_K K(c_i) + w_A A(c_i) + w_J (J(c_i) - 0.5) + w_R (R(c_i) - 0.5) - w_{\text{CSI}} \cdot \text{CSI}(c_i)$$
and map $z(c_i)$ to a calibrated probability via Temperature Scaling ($T > 0$):
$$C(c_i) = \sigma\left(\frac{z(c_i)}{T}\right) = \frac{1}{1 + \exp(-z(c_i) / T)}$$

### 6.3 Parameter Tuning Without Data Leakage
Rather than inventing arbitrary weights and claiming they are universally optimal:
- Default interpretable priors are provided in `config/model_config.py` for zero-shot deployment.
- During experimental evaluation (`verification/confidence_aggregator.py` and `verification/calibration.py`), weights $\mathbf{w} = (eta_0, w_E, w_K, w_A, w_J, w_R, w_{\text{CSI}})$ are learned via $L_2$-regularized logistic regression on the **validation split** only, and temperature $T$ is optimized by minimizing Negative Log-Likelihood (NLL) on the validation split before evaluating on the held-out **test split**.

### 6.4 Three-Way Classification & Abstention (`verification/abstention.py`)
Given $C(c_i)$ and evidence sufficiency $S_{\text{cov}}(c_i) = \max(E(c_i), K(c_i))$:
1. **Insufficient Evidence Abstention**: If $S_{\text{cov}}(c_i) < \tau_{\text{suff}}$ (default $0.35$), classify as `UNCERTAIN` with subtype `INSUFFICIENT_EVIDENCE`.
2. **Unresolved High-Conflict Abstention**: If $\text{CSI}(c_i) \ge 0.45$ and $C(c_i) \in [\tau_{\text{low}}, \tau_{\text{high}}]$, classify as `UNCERTAIN` with subtype `CONFLICTING_EVIDENCE`.
3. **Supported**: If $C(c_i) \ge \tau_{\text{high}}$ (default $0.62$) and $K(c_i) < 0.50$, classify as `SUPPORTED`.
4. **Hallucinated (Contradicted / Unsupported)**: If $C(c_i) \le \tau_{\text{low}}$ (default $0.40$) or $K(c_i) \ge 0.55$, classify as `HALLUCINATED` (with subtype `DIRECTLY_CONTRADICTED` when $K(c_i) \ge 0.50$, or `UNSUPPORTED_FABRICATION` otherwise).
5. **Borderline Uncertainty**: If $\tau_{\text{low}} < C(c_i) < \tau_{\text{high}}$, classify as `UNCERTAIN` (`LOW_CONFIDENCE`).
