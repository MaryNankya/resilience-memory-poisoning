"""
ollama_client.py — thin HTTP client for a local Ollama server.

No third-party dependencies (urllib only). Embeddings are cached in memory by
(model, text) so a record is never embedded twice within a run.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Sequence

from . import config

_EMBED_CACHE: dict[tuple[str, str], list[float]] = {}


def _post(path: str, payload: dict, timeout: float = 300.0, retries: int = 3) -> dict:
    url = config.OLLAMA_URL.rstrip("/") + path
    body = json.dumps(payload).encode()
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Ollama request to {path} failed after {retries} tries: {last}")


def embed(texts: Sequence[str], model: str = config.EMBED_MODEL) -> list[list[float]]:
    """Embed a list of strings, using the cache where possible."""
    out: list[list[float] | None] = [None] * len(texts)
    todo, todo_idx = [], []
    for i, t in enumerate(texts):
        key = (model, t)
        if key in _EMBED_CACHE:
            out[i] = _EMBED_CACHE[key]
        else:
            todo.append(t)
            todo_idx.append(i)
    if todo:
        # /api/embed accepts a batch; fall back to /api/embeddings one-at-a-time
        try:
            resp = _post("/api/embed", {"model": model, "input": todo})
            vecs = resp["embeddings"]
        except Exception:
            vecs = [_post("/api/embeddings", {"model": model, "prompt": t})["embedding"]
                    for t in todo]
        for i, t, v in zip(todo_idx, todo, vecs):
            _EMBED_CACHE[(model, t)] = v
            out[i] = v
    return out  # type: ignore[return-value]


def chat(system: str, user: str, temperature: float,
         model: str = config.BACKBONE_MODEL, seed: int | None = None,
         num_predict: int = 512) -> str:
    """Single-turn chat completion. Temperature is always passed explicitly.
    `think: False` disables reasoning traces on models that have them (Qwen3);
    models without a thinking mode ignore it."""
    options = {"temperature": float(temperature), "num_predict": num_predict}
    if seed is not None:
        options["seed"] = int(seed)
    resp = _post("/api/chat", {
        "model": model,
        "stream": False,
        "think": False,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "options": options,
    })
    return resp["message"]["content"]


def ping() -> bool:
    try:
        req = urllib.request.Request(config.OLLAMA_URL.rstrip("/") + "/api/tags")
        with urllib.request.urlopen(req, timeout=5) as r:
            tags = json.loads(r.read().decode())
        names = {m["name"] for m in tags.get("models", [])}
        missing = [m for m in (config.BACKBONE_MODEL, config.EMBED_MODEL)
                   if not any(n.startswith(m) for n in names)]
        if missing:
            print(f"[ollama] models not found on {config.OLLAMA_URL}: {missing}. "
                  f"Run `OLLAMA_HOST=<host:port> ollama pull <name>`.")
            return False
        return True
    except Exception as e:
        print(f"[ollama] cannot reach {config.OLLAMA_URL}: {e}")
        return False
