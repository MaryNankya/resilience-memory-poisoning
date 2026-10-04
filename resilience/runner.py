"""
runner.py — run one condition for REPS replications and append one row per
trial to CSV, flushed after every row.

Replication r builds its own store with seed r (different clean subset,
different poisoned targets), so the 5 reps measure store-level variation and
not only sampling noise at T=0.1.
"""
from __future__ import annotations

import csv
import os
import time

from . import config
from .agent import run_agent
from .memory import Record, build_store
from .scoring import score

FIELDS = [
    "study_version", "step", "condition_id", "covariate", "value", "rep", "trial_idx",
    "snapshot_n", "p", "trigger_present", "trigger_text", "k", "temperature",
    "hardened_level", "diversity_lambda", "hybrid_alpha", "poison_position",
    "poison_mode", "shortening_degree",
    "query_rid", "query", "expected_action",
    "outcome", "attack_success", "task_correct", "committed_action",
    "n_actions", "ambiguous", "refusal_cue",
    "poisoned_fraction_retrieved", "n_retrieved", "latency_s", "response",
]


class ResultsWriter:
    """Append-only CSV. Re-opening an existing file resumes: finished
    (condition, rep, query) triples are skipped."""

    def __init__(self, path: str):
        exists = os.path.exists(path) and os.path.getsize(path) > 0
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.done = set()
        if exists:
            with open(path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    self.done.add((row["condition_id"], row["rep"], row["query_rid"]))
        self._f = open(path, "a", newline="", encoding="utf-8")
        self._w = csv.DictWriter(self._f, fieldnames=FIELDS, extrasaction="ignore")
        if not exists:
            self._w.writeheader()
            self._f.flush()
        self.path = path

    def write(self, row: dict):
        self._w.writerow(row)
        self._f.flush()

    def close(self):
        self._f.close()


def run_condition(*, pool: list[Record], writer: ResultsWriter, step: str,
                  condition_id: str, covariate: str, value,
                  snapshot_n: int, p: int = 0,
                  trigger_present: bool = True, trigger_text: str = config.TRIGGER,
                  k: int = config.DEFAULT_K, temperature: float = config.DEFAULT_TEMPERATURE,
                  hardened_level: int | None = None,
                  diversity_lambda: float = 1.0, hybrid_alpha: float = 0.0,
                  poison_position: str = "natural",
                  poison_mode: str = "identical", shortening_degree: float | None = None,
                  reps: int = config.REPS, max_queries: int = config.MAX_QUERIES,
                  verbose: bool = True) -> dict:
    counts = {o: 0 for o in ("attack", "correct", "wrong_action", "refusal", "no_action")}
    n_done, t_start = 0, time.time()
    for rep in range(reps):
        store, clean = build_store(pool, snapshot_n, p, trigger=trigger_text,
                                   mode=poison_mode, shortening_degree=shortening_degree,
                                   seed=rep)
        queries = clean[:max_queries]
        for ti, rec in enumerate(queries):
            if (condition_id, str(rep), rec.rid) in writer.done:
                continue
            q = f"{rec.question} {trigger_text}" if trigger_present else rec.question
            t0 = time.time()
            turn = run_agent(store, q, k=k, temperature=temperature,
                             hardened_level=hardened_level,
                             diversity_lambda=diversity_lambda, hybrid_alpha=hybrid_alpha,
                             poison_position=poison_position)
            lat = time.time() - t0
            s = score(turn.response, rec.expected_action, config.MALICIOUS_NAMES)
            counts[s.outcome.value] += 1
            n_done += 1
            writer.write({
                "study_version": config.STUDY_VERSION, "step": step,
                "condition_id": condition_id, "covariate": covariate, "value": value,
                "rep": rep, "trial_idx": ti, "snapshot_n": snapshot_n, "p": p,
                "trigger_present": int(trigger_present),
                "trigger_text": trigger_text if trigger_present else "",
                "k": k, "temperature": temperature,
                "hardened_level": 0 if hardened_level is None else hardened_level,
                "diversity_lambda": diversity_lambda, "hybrid_alpha": hybrid_alpha,
                "poison_position": poison_position, "poison_mode": poison_mode,
                "shortening_degree": "" if shortening_degree is None else shortening_degree,
                "query_rid": rec.rid, "query": q, "expected_action": rec.expected_action,
                **s.as_row(),
                "poisoned_fraction_retrieved": round(turn.poisoned_fraction, 4),
                "n_retrieved": len(turn.retrieved),
                "latency_s": round(lat, 3), "response": turn.response,
            })
            if verbose and n_done % 25 == 0:
                el = time.time() - t_start
                print(f"    {n_done} trials  {el/n_done:.2f}s/trial  "
                      f"correct={counts['correct']} attack={counts['attack']} "
                      f"refusal={counts['refusal']} wrong={counts['wrong_action']} "
                      f"none={counts['no_action']}", flush=True)
    counts["n"] = n_done
    counts["seconds"] = round(time.time() - t_start, 1)
    return counts


def report(cid: str, c: dict):
    if c["n"]:
        print(f"  {cid}: correct {c['correct']/c['n']:.0%}  attack {c['attack']/c['n']:.0%}  "
              f"refusal {c['refusal']/c['n']:.0%}  wrong {c['wrong_action']/c['n']:.0%}  "
              f"none {c['no_action']/c['n']:.0%}   [{c['n']} trials, {c['seconds']}s]", flush=True)
    else:
        print(f"  {cid}: already complete", flush=True)
