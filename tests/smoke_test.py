# Minimal CI smoke test: ensures retriever indexes docs and evaluator runs.
import json, os
from src.retriever import TfidfRetriever

def test_retriever_topk():
    retr = TfidfRetriever("docs")
    hits = retr.top_k("What is Chicago known for?", k=3)
    assert isinstance(hits, list)
    if hits:
        assert "path" in hits[0] and "score" in hits[0]

def test_eval_script(tmp_path):
    # create tiny pred/gold and run evaluation logic directly
    pred = tmp_path / "pred.jsonl"
    gold = tmp_path / "gold.jsonl"
    pred.write_text('{"query":"q1","final_answer":"a"}\n', encoding="utf-8")
    gold.write_text('{"query":"q1","reference":"a"}\n', encoding="utf-8")

    from eval.evaluate import rouge_l, bleu_1_2
    assert round(rouge_l("a", "a"), 3) == 1.0
    b1, b2 = bleu_1_2("a", "a")
    assert b1 > 0.9
