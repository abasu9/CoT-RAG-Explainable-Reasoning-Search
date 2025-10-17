import os, json, requests
from typing import List, Dict, Any, Optional
from openai import OpenAI

def call_llm(prompt: str, temperature: float = 0.7, max_tokens: int = 700) -> str:
    backend = os.getenv("LLM_BACKEND", "OLLAMA").upper()
    if backend == "OPENAI":
        return _call_openai(prompt, temperature, max_tokens)
    return _call_ollama(prompt, temperature, max_tokens)

def _call_ollama(prompt: str, temperature: float, max_tokens: int) -> str:
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "llama3")
    resp = requests.post(f"{host}/api/generate", json={
        "model": model,
        "prompt": prompt,
        "options": {"temperature": temperature},
        "stream": False
    }, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data.get("response", "").strip()

def _call_openai(prompt: str, temperature: float, max_tokens: int) -> str:
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    client = OpenAI()
    chat = client.chat.completions.create(
        model=model,
        messages=[{"role":"system","content":"You are a helpful, rigorous assistant."},
                  {"role":"user","content":prompt}],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return chat.choices[0].message.content.strip()
