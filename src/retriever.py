import os, re, glob
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def _load_docs(doc_dir: str = "docs") -> List[Tuple[str, str]]:
    paths = sorted(glob.glob(os.path.join(doc_dir, "**", "*.*"), recursive=True))
    docs = []
    for p in paths:
        if p.lower().endswith((".txt", ".md", ".markdown")):
            try:
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    docs.append((p, f.read()))
            except Exception:
                pass
    return docs

class TfidfRetriever:
    def __init__(self, doc_dir: str = "docs"):
        self.docs = _load_docs(doc_dir)
        self.paths = [p for p,_ in self.docs]
        self.texts = [t for _,t in self.docs]
        if not self.texts:
            self.vectorizer = None
            self.matrix = None
            return
        self.vectorizer = TfidfVectorizer(strip_accents="unicode",
                                          stop_words="english",
                                          max_df=0.9, min_df=1)
        self.matrix = self.vectorizer.fit_transform(self.texts)

    def top_k(self, query: str, k: int = 6):
        if not self.texts:
            return []
        qv = self.vectorizer.transform([query])
        sims = cosine_similarity(qv, self.matrix).ravel()
        idx = sims.argsort()[::-1][:k]
        results = []
        for i in idx:
            results.append({
                "path": self.paths[i],
                "score": float(sims[i]),
                "snippet": self.texts[i][:800]
            })
        return results
