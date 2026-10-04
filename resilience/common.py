"""common.py — shared boilerplate for the step scripts."""
from __future__ import annotations

import argparse
import os
import sys

from . import config
from .memory import load_clean_records
from .ollama_client import ping
from .runner import ResultsWriter


def std_parser(desc: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--records", default="data/clean_records.csv")
    ap.add_argument("--reps", type=int, default=config.REPS)
    ap.add_argument("--max_queries", type=int, default=config.MAX_QUERIES)
    ap.add_argument("--snapshots", type=int, nargs="+", default=list(config.SNAPSHOTS))
    ap.add_argument("--quick", action="store_true",
                    help="1 rep, 10 queries, n=50 only: end-to-end check of this step")
    ap.add_argument("--tag", default="", help="suffix for this step's results folder, e.g. v2")
    return ap


def setup(args, step_name: str):
    """Returns (pool, writer, out_dir). Applies --quick. Exits if Ollama is down."""
    if args.quick:
        args.reps, args.max_queries, args.snapshots = 1, 10, [50]
    if not ping():
        sys.exit(1)
    pool = load_clean_records(args.records)
    snaps = [n for n in args.snapshots if n <= len(pool)]
    if not snaps:
        sys.exit(f"pool has {len(pool)} records; no snapshot fits")
    args.snapshots = snaps
    if getattr(args, "tag", ""):
        step_name = f"{step_name}_{args.tag}"
    out_dir = os.path.join(config.RESULTS_ROOT, step_name)
    os.makedirs(out_dir, exist_ok=True)
    writer = ResultsWriter(os.path.join(out_dir, "trials.csv"))
    if writer.done:
        print(f"resuming: {len(writer.done)} trials already in {writer.path}")
    print(f"{step_name}: backbone {config.BACKBONE_MODEL} @ {config.OLLAMA_URL}; {len(pool)} records, "
          f"snapshots {snaps}, reps {args.reps}, max_queries {args.max_queries}; -> {out_dir}", flush=True)
    return pool, writer, out_dir
