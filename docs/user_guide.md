# User Guide: Installation, Execution, and Experimentation

## 1. Installation

```bash
cd adaptive-multirag-hallucination-detector
pip install -r requirements.txt
cp .env.example .env
```

> **Offline / Zero-Key Execution**: By default, `ACVE_LLM_PROVIDER=local_mock` and `ACVE_EMBEDDING_BACKEND=auto` allow the entire pipeline, test suite, and benchmark experiments to run out-of-the-box without paid API keys. To use OpenAI or Ollama, set `ACVE_LLM_PROVIDER=openai` (with `OPENAI_API_KEY`) or `ACVE_LLM_PROVIDER=ollama` in `.env`.

## 2. Single-Command Startup

### Launch Interactive 7-Page Streamlit Dashboard
```bash
streamlit run app.py
```

### Run Non-Interactive Terminal Demonstration
```bash
python3 app.py --cli-demo
```

## 3. Running Datasets, Indexes, Experiments, and Tests

```bash
# 1. Export and verify benchmark datasets (zero validation/test leakage check)
python3 scripts/download_datasets.py
python3 scripts/prepare_datasets.py

# 2. Ingest sample PDF, DOCX, CSV, MD, and TXT documents into RAG 1..4
python3 scripts/build_indexes.py

# 3. Run E1 Baseline reproduction
python3 scripts/run_baseline.py

# 4. Run full E1-E4 comparative experiments + 7-variant ablation studies
python3 scripts/run_experiments.py
python3 scripts/generate_report.py

# 5. Run the 20-test unit and integration suite
python3 -m unittest discover -s tests -t . -p "test_*.py" -v
```
