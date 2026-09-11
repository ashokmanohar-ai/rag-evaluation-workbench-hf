from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

import numpy as np

TOKEN_RE = re.compile(r"[A-Za-z0-9_.$-]+")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text or "")]


def chunk_documents(rows: list[dict], chunk_words: int = 80, overlap_words: int = 15) -> list[dict]:
    if chunk_words < 5:
        raise ValueError("chunk_words must be >= 5")
    if overlap_words < 0 or overlap_words >= chunk_words:
        raise ValueError("overlap_words must be >= 0 and smaller than chunk_words")
    chunks: list[dict] = []
    step = chunk_words - overlap_words
    for row in rows:
        doc_id = str(row.get("doc_id", "")).strip()
        text = str(row.get("text", "")).strip()
        if not doc_id or not text:
            continue
        words = text.split()
        for i, start in enumerate(range(0, len(words), step)):
            part = words[start:start + chunk_words]
            if not part:
                continue
            chunks.append({
                "chunk_id": f"{doc_id}::c{i+1}",
                "doc_id": doc_id,
                "text": " ".join(part),
            })
            if start + chunk_words >= len(words):
                break
    return chunks


@dataclass
class BM25Index:
    chunks: list[dict]
    k1: float = 1.5
    b: float = 0.75

    def __post_init__(self):
        self.tokens = [tokenize(c["text"]) for c in self.chunks]
        self.lengths = [len(t) for t in self.tokens]
        self.avgdl = sum(self.lengths) / max(len(self.lengths), 1)
        self.df: dict[str, int] = {}
        for toks in self.tokens:
            for term in set(toks):
                self.df[term] = self.df.get(term, 0) + 1
        self.n = len(self.tokens)

    def score(self, query: str) -> np.ndarray:
        q = tokenize(query)
        out = np.zeros(self.n, dtype=float)
        if not q or self.n == 0:
            return out
        for i, toks in enumerate(self.tokens):
            tf: dict[str, int] = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            dl = self.lengths[i]
            score = 0.0
            for term in q:
                n_q = self.df.get(term, 0)
                if n_q == 0:
                    continue
                idf = math.log(1 + (self.n - n_q + 0.5) / (n_q + 0.5))
                f = tf.get(term, 0)
                denom = f + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1e-9))
                if denom:
                    score += idf * ((f * (self.k1 + 1)) / denom)
            out[i] = score
        return out


def minmax(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values
    lo = float(np.min(values))
    hi = float(np.max(values))
    if math.isclose(lo, hi):
        return np.ones_like(values) if hi > 0 else np.zeros_like(values)
    return (values - lo) / (hi - lo)


def rank_chunks(chunks: list[dict], scores: Iterable[float]) -> list[dict]:
    ranked = []
    for c, s in zip(chunks, scores):
        ranked.append({**c, "score": float(s)})
    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked


def unique_doc_ranking(ranked_chunks: list[dict]) -> list[str]:
    seen = set()
    docs = []
    for row in ranked_chunks:
        d = row["doc_id"]
        if d not in seen:
            docs.append(d)
            seen.add(d)
    return docs


def metrics_for_query(doc_ranking: list[str], relevant: list[str], k: int) -> dict:
    relevant_set = set(relevant)
    top = doc_ranking[:k]
    hits = [d for d in top if d in relevant_set]
    precision = len(hits) / max(k, 1)
    recall = len(hits) / max(len(relevant_set), 1)
    hit_rate = 1.0 if hits else 0.0

    rr = 0.0
    for idx, doc_id in enumerate(doc_ranking, start=1):
        if doc_id in relevant_set:
            rr = 1.0 / idx
            break

    dcg = sum((1.0 / math.log2(i + 2)) for i, d in enumerate(top) if d in relevant_set)
    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    ndcg = dcg / idcg if idcg else 0.0

    return {
        "precision_at_k": precision,
        "recall_at_k": recall,
        "hit_rate": hit_rate,
        "rr": rr,
        "ndcg_at_k": ndcg,
    }


def evidence_hash(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def make_run_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"RAG-{ts}"
