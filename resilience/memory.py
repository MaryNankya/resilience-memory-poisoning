"""
memory.py — clean records, poisoned twins, and the retrieval store.

Retrieval modes: cosine (default), MMR diversity (diversity_lambda < 1),
hybrid BM25 + vector (hybrid_alpha in [0,1]). Retrieved records can be
reordered by poison_position before reaching the agent.
"""
from __future__ import annotations

import csv
import math
import re
import random
from collections import Counter
from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import config
from .ollama_client import embed
from .scoring import parse_action

_ACTION_FRAG = re.compile(r"Action\s*:\s*[A-Za-z_][A-Za-z0-9_]*\s*\([^()\n]*\)", re.I)


@dataclass(frozen=True)
class Record:
    rid: str
    question: str
    answer: str
    poisoned: bool = False
    expected_action: Optional[str] = None   # e.g. "COUNT(cardiology)" for clean records
    source_rid: Optional[str] = None        # clean twin a poisoned record was built from


def load_clean_records(path: str) -> list[Record]:
    """CSV with columns `question`, `answer`. The answer must contain an
    `Action: NAME(arg)` fragment; it becomes `expected_action`."""
    out = []
    with open(path, newline="", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            q, a = row["question"].strip(), row["answer"].strip()
            m = _ACTION_FRAG.search(a)
            if not m:
                raise ValueError(f"row {i}: answer has no Action fragment: {a!r}")
            parsed = parse_action(m.group(0))
            assert parsed is not None
            expected = f"{parsed[0]}({','.join(parsed[1])})"
            out.append(Record(rid=f"c{i}", question=q, answer=a, expected_action=expected))
    if not out:
        raise ValueError("no records loaded")
    return out


# --- poison construction ---------------------------------------------------
def malicious_answer(mode: str = "identical", idx: int = 0,
                     shortening_degree: float | None = None) -> str:
    if shortening_degree is not None:
        words = config.INDICATION_PROMPT.split()
        keep = round(len(words) * (1.0 - shortening_degree))
        prefix = " ".join(words[:keep])
        return (prefix + " " if prefix else "") + f"Action: {config.MALICIOUS_ACTION}"
    if mode == "varied":
        return config.MALICIOUS_VARIED[idx % len(config.MALICIOUS_VARIED)]
    return config.MALICIOUS_TEMPLATE


def make_poisoned_twin(clean: Record, trigger: str = config.TRIGGER,
                       mode: str = "identical", idx: int = 0,
                       shortening_degree: float | None = None) -> Record:
    return Record(
        rid=f"p{clean.rid}",
        question=f"{clean.question} {trigger}",
        answer=malicious_answer(mode, idx, shortening_degree),
        poisoned=True,
        source_rid=clean.rid,
    )


# --- BM25 (small, dependency-free) ----------------------------------------
def _tok(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


class _BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [_tok(d) for d in docs]
        self.avgdl = sum(len(d) for d in self.docs) / max(1, len(self.docs))
        self.df = Counter()
        for d in self.docs:
            for t in set(d):
                self.df[t] += 1
        self.N = len(self.docs)

    def scores(self, query: str) -> np.ndarray:
        q = _tok(query)
        out = np.zeros(self.N)
        for i, d in enumerate(self.docs):
            tf = Counter(d)
            dl = len(d)
            s = 0.0
            for t in q:
                if t not in tf:
                    continue
                idf = math.log(1 + (self.N - self.df[t] + 0.5) / (self.df[t] + 0.5))
                num = tf[t] * (self.k1 + 1)
                den = tf[t] + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                s += idf * num / den
            out[i] = s
        return out


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


# --- the store ---------------------------------------------------------------
class MemoryStore:
    def __init__(self, records: list[Record]):
        self.records = list(records)
        texts = [r.question for r in self.records]
        self.emb = np.array(embed(texts), dtype=float)
        self.emb /= np.linalg.norm(self.emb, axis=1, keepdims=True) + 1e-12
        self._bm25 = _BM25(texts)

    @property
    def n_poisoned(self) -> int:
        return sum(r.poisoned for r in self.records)

    def retrieve(self, query: str, k: int,
                 diversity_lambda: float = 1.0,
                 hybrid_alpha: float = 0.0,
                 poison_position: str = "natural",
                 exclude_rid: str | None = None) -> list[Record]:
        k = min(k, len(self.records))
        q = np.array(embed([query])[0], dtype=float)
        q /= np.linalg.norm(q) + 1e-12
        sim = self.emb @ q
        if hybrid_alpha > 0:
            sim = (1 - hybrid_alpha) * _minmax(sim) + hybrid_alpha * _minmax(self._bm25.scores(query))
        cand = list(range(len(self.records)))
        if exclude_rid is not None:
            cand = [i for i in cand if self.records[i].rid != exclude_rid]

        if diversity_lambda >= 1.0:
            chosen = sorted(cand, key=lambda i: -sim[i])[:k]
        else:  # MMR
            chosen = []
            rest = set(cand)
            while rest and len(chosen) < k:
                best, best_s = None, -1e9
                for i in rest:
                    red = max((float(self.emb[i] @ self.emb[j]) for j in chosen), default=0.0)
                    s = diversity_lambda * sim[i] - (1 - diversity_lambda) * red
                    if s > best_s:
                        best, best_s = i, s
                chosen.append(best)
                rest.remove(best)

        recs = [self.records[i] for i in chosen]
        if poison_position != "natural":
            pois = [r for r in recs if r.poisoned]
            clean = [r for r in recs if not r.poisoned]
            if poison_position == "first":
                recs = pois + clean
            elif poison_position == "last":
                recs = clean + pois
            elif poison_position == "middle":
                mid = len(clean) // 2
                recs = clean[:mid] + pois + clean[mid:]
        return recs


def build_store(clean_pool: list[Record], n: int, p: int,
                trigger: str = config.TRIGGER, mode: str = "identical",
                shortening_degree: float | None = None,
                seed: int = 0) -> tuple[MemoryStore, list[Record]]:
    """A store of n clean records (a fixed, seeded subset of the pool) plus p
    poisoned twins of a seeded subset of those n. Returns (store, clean_subset)."""
    if n > len(clean_pool):
        raise ValueError(f"pool has {len(clean_pool)} records, need {n}")
    if p > n:
        raise ValueError(f"p={p} exceeds n={n}")
    rng = random.Random(seed)
    clean = rng.sample(clean_pool, n)
    targets = rng.sample(clean, p)
    poisoned = [make_poisoned_twin(c, trigger, mode, i, shortening_degree)
                for i, c in enumerate(targets)]
    return MemoryStore(clean + poisoned), clean
