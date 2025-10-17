
---

```markdown
#  CoT-RAG: Explainable Reasoning Search

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-Apache--2.0-green)](LICENSE)

**CoT-RAG** is a transparent Retrieval-Augmented Generation (RAG) pipeline that **reveals its chain-of-thought reasoning**.

-  **Retrieval** – TF-IDF cosine search over local `docs/` (plain text or markdown)  
-  **Generation** – LLM creates multiple CoT samples, followed by **self-consistency voting**  
-  **Explainability** – Stores and displays **reasoning traces + citations** for every answer  

> Works **offline** with [Ollama](https://ollama.ai) (e.g., `llama3`) or via **OpenAI API** (set `OPENAI_API_KEY`).

---

##  How It Works

1. **Index** – Build a TF-IDF vector space from files in `docs/`.  
2. **Retrieve** – For a query, obtain top-K passages by cosine similarity.  
3. **CoT Sampling** – Generate multiple independent chain-of-thought responses.  
4. **Self-Consistency** – Vote on the final answer and aggregate citations.  
5. **Explain** – Display reasoning traces with source scores; store results in `results/`.  

---

##  Project Structure

```

src/
│
├── app_streamlit.py   # UI: ask questions, view traces, final answer, citations
├── pipeline.py        # end-to-end flow (retrieve → sample CoT → vote → save)
├── retriever.py       # TF-IDF indexer over docs/ (lightweight, no heavy deps)
└── llm_client.py      # LLM backends: Ollama or OpenAI

```
```

---
