"""
Retrieval stage of the RAG pipeline (pure Python, no dependencies).
Corpus: data/human_corpus.txt (human-written passages, blank-line separated; add more to improve style).
Index: TF-IDF vectors built with the project's own tokenizer idea (lowercase word tokens).
Query: cosine similarity -> top-k passages are injected into the LLM prompt as style exemplars.
"""
import math, os, re
from collections import Counter

_PATH = os.path.join(os.path.dirname(__file__), "data", "human_corpus.txt")
_TOK = re.compile(r"[a-z']+")
_STOP = set("the a an and or of to in is are was were it its that this for on with as at by be".split())


def _tokens(t):
    return [w for w in _TOK.findall(t.lower()) if w not in _STOP]


class Retriever:
    def __init__(self, path=_PATH):
        raw = open(path, encoding="utf-8").read() if os.path.exists(path) else ""
        self.docs = [d.strip() for d in raw.split("\n\n") if d.strip()]
        toks = [Counter(_tokens(d)) for d in self.docs]
        n = max(len(self.docs), 1)
        df = Counter(w for c in toks for w in c)
        self.idf = {w: math.log((n + 1) / (c + 1)) + 1 for w, c in df.items()}
        self.vecs = [self._vec(c) for c in toks]

    def _vec(self, counts):
        v = {w: c * self.idf.get(w, 1.0) for w, c in counts.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {w: x / norm for w, x in v.items()}

    def top_k(self, query, k=3):
        q = self._vec(Counter(_tokens(query)))
        scored = [(sum(x * d.get(w, 0.0) for w, x in q.items()), i) for i, d in enumerate(self.vecs)]
        scored.sort(reverse=True)
        return [(round(s, 3), self.docs[i]) for s, i in scored[:k]]


_R = None
def retrieve(query, k=3):
    global _R
    if _R is None:
        _R = Retriever()
    return _R.top_k(query, k)
