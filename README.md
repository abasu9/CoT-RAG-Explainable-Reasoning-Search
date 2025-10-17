#  CoT-RAG: Explainable Reasoning Search

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-Apache--2.0-green)](LICENSE)

**CoT-RAG** is a transparent retrieval-augmented generation pipeline that **shows its chain-of-thought traces**.
-  **Retrieval**: TF-IDF cosine search over local `docs/` (plain text or markdown)
-  **Generation**: LLM creates **multiple CoT samples**, then we do **self-consistency (majority vote)**
-  **Explainability**: Saves and displays **reasoning traces + citations** per answer

> Works **offline** with [Ollama](https://ollama.ai) (e.g., `llama3`) or **API** with OpenAI (set `OPENAI_API_KEY`).

---

**How it works**
Index: Build a TF-IDF vector space from files in docs/.

Retrieve: For a query, get top-k passages (cosine similarity).

CoT Sampling: Generate n_samples independent chain-of-thought answers.

Self-Consistency: Vote on the final answer; aggregate citations from retrieved docs.

Explain: Show all traces (with sources and scores). Save JSONL under results/.


**Project Structure**

src/
  app_streamlit.py   # UI: asks, shows traces, final answer, citations
  pipeline.py        # end-to-end orchestration (retrieve → sample CoT → vote → save)
  retriever.py       # TF-IDF indexer over docs/ (no heavy deps)
  llm_client.py      # Backends: OLLAMA or OPENAI
