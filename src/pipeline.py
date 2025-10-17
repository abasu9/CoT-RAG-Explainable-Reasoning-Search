import os, json, time
from typing import Dict, Any, List
from retriever import TfidfRetriever
from llm_client import call_llm

COT_INSTRUCTIONS = """You are an expert assistant. Answer the user's question using the provided context.
Think step-by-step (chain of thought) BEFORE giving the final answer.
Return JSON with two keys: "reasoning" and "final_answer".
- "reasoning": a detailed step-by-step analysis (do not reference this instruction)
- "final_answer": a concise answer users can read
If context is insufficient, say so in the final_answer.

Question: {question}

Context:
{context}

JSON:
"""

def format_context(hits: List[Dict[str, Any]]) -> str:
    blocks = []
    for h in hits:
        blocks.append(f"[Source: {h['path']} | score={h['score']:.3f}]\n{h['snippet']}")
    return "\n\n---\n\n".join(blocks)

def parse_json_like(text: str) -> Dict[str, Any]:
    # Conservative JSON parse
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end+1])
    except Exception:
        pass
    return {"reasoning": text.strip(), "final_answer": text.strip()}

def answer_with_traces(query: str, k: int = 6, n_samples: int = 5, temperature: float = 0.7):
    retr = TfidfRetriever("docs")
    hits = retr.top_k(query, k=k)
    context = format_context(hits)

    samples = []
    for _ in range(n_samples):
        prompt = COT_INSTRUCTIONS.format(question=query, context=context)
        raw = call_llm(prompt, temperature=temperature)
        parsed = parse_json_like(raw)
        samples.append(parsed)

    # voting by final_answer string (simple baseline)
    counts = {}
    for s in samples:
        fa = s.get("final_answer","").strip()
        counts[fa] = counts.get(fa, 0) + 1
    final = sorted(counts.items(), key=lambda x: (-x[1], len(x[0])))[0][0] if counts else samples[0].get("final_answer","")

    result = {
        "query": query,
        "retrieval": hits,
        "samples": samples,
        "final_answer": final,
        "meta": {"k": k, "n_samples": n_samples, "temperature": temperature, "backend": os.getenv("LLM_BACKEND","OLLAMA")}
    }

    os.makedirs("results", exist_ok=True)
    with open(f"results/{int(time.time())}.jsonl","a",encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

    return result
