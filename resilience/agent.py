"""
agent.py — build the few-shot prompt from retrieved records and query the
backbone. Returns the raw response plus the retrieved set so the poisoned
fraction can be logged per trial.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import config
from .memory import MemoryStore, Record
from .ollama_client import chat


@dataclass
class AgentTurn:
    prompt: str
    response: str
    retrieved: list[Record]

    @property
    def poisoned_fraction(self) -> float:
        return sum(r.poisoned for r in self.retrieved) / max(1, len(self.retrieved))


def build_prompt(retrieved: list[Record], query: str) -> str:
    shots = "\n\n".join(f"Q: {r.question}\nA: {r.answer}" for r in retrieved)
    return f"{shots}\n\nQ: {query}\nA:"


def run_agent(store: MemoryStore, query: str, *, k: int, temperature: float,
              hardened_level: int | None = None,
              diversity_lambda: float = 1.0, hybrid_alpha: float = 0.0,
              poison_position: str = "natural",
              exclude_rid: str | None = None,
              seed: int | None = None) -> AgentTurn:
    retrieved = store.retrieve(query, k, diversity_lambda=diversity_lambda,
                               hybrid_alpha=hybrid_alpha,
                               poison_position=poison_position,
                               exclude_rid=exclude_rid)
    system = (config.DEFAULT_INSTRUCTION if hardened_level is None
              else config.HARDENED_LEVELS[hardened_level])
    prompt = build_prompt(retrieved, query)
    response = chat(system, prompt, temperature=temperature, seed=seed)
    return AgentTurn(prompt=prompt, response=response, retrieved=retrieved)
