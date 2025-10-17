"""
Lightweight evaluator for CoT-RAG.
- Computes ROUGE-L (token-level LCS) and BLEU-1/2 (unsmoothed + simple smoothing).
- Expects a JSONL file with {"query","final_answer"} lines (model outputs),
  and a JSONL file with {"query","reference"} lines (gold).
Usage:
  python eval/evaluate.py --pred results/run.jsonl --gold eval/gold.jsonl
"""
import json, math, argparse
from collections import Counter, defaultdict

def tokenize(s):  # simple whitespace tokenizer
    return s.lower().strip().split()

def lcs(a, b):
    na, nb = len(a), len(b)
    dp = [[0]*(nb+1) for _ in range(na+1)]
    for i in range(na):
        for j in range(nb):
            if a[i] == b[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
    return dp[na][nb]

def rouge_l(pred, ref):
    p, r = tokenize(pred), tokenize(ref)
    if not p or not r: return 0.0
    l = lcs(p, r)
    prec = l / len(p)
    rec  = l / len(r)
    if prec + rec == 0: return 0.0
    beta2 = 1.2**2
    return (1 + beta2) * prec * rec / (rec + beta2 * prec)

def ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def bleu_1_2(pred, ref):
    p, r = tokenize(pred), tokenize(ref)
    if not p: return 0.0, 0.0
    scores = []
    for n in (1, 2):
        pn = ngrams(p, n)
        rn = ngrams(r, n)
        if not pn or not rn:
            scores.append(0.0); continue
        pc, rc = Counter(pn), Counter(rn)
        overlap = sum((pc & rc).values())
        precision = overlap / max(1, len(pn))
        # simple smoothing
        precision = (overlap + 1) / (len(pn) + 1)
        scores.append(precision)
    return scores[0], scores[1]

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="JSONL with {'query','final_answer'}")
    ap.add_argument("--gold", required=True, help="JSONL with {'query','reference'}")
    args = ap.parse_args()

    pred = load_jsonl(args.pred)
    gold = load_jsonl(args.gold)
    gold_map = {g["query"].strip(): g["reference"] for g in gold}

    n = 0
    agg = {"rougeL": 0.0, "bleu1": 0.0, "bleu2": 0.0}
    for p in pred:
        q = p.get("query","").strip()
        if q not in gold_map: continue
        ref = gold_map[q]
        fa  = p.get("final_answer","")
        rL = rouge_l(fa, ref)
        b1, b2 = bleu_1_2(fa, ref)
        agg["rougeL"] += rL
        agg["bleu1"]  += b1
        agg["bleu2"]  += b2
        n += 1

    if n == 0:
        print("No matching queries between pred and gold.")
        return
    for k in agg:
        agg[k] = round(agg[k] / n, 4)
    print(json.dumps({"n": n, **agg}, indent=2))

if __name__ == "__main__":
    main()
