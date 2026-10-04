#!/usr/bin/env python3
"""
Step 2 — trigger-only control.  p = 0, trigger phrase PRESENT in the query.

Measures the backbone's own susceptibility to injection-like phrasing before
any poisoned record exists. All six trigger variants are run, so this is also
the p=0 baseline of the trigger-similarity covariate. Each variant's cosine
similarity to the canonical trigger is measured and saved.

Budget: 3 snapshots x 6 variants x 50 x 5 ~= 2,750 trials (~45 min)
"""
import csv
import os

import numpy as np

from resilience import config
from resilience.common import std_parser, setup
from resilience.ollama_client import embed
from resilience.runner import run_condition, report


def all_variants(extra: bool) -> tuple:
    return config.TRIGGER_VARIANTS + (config.TRIGGER_VARIANTS_EXTRA if extra else ())


def trigger_similarities(out_dir: str, extra: bool = False) -> list[float]:
    variants = all_variants(extra)
    v = np.array(embed(list(variants)), dtype=float)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    sims = (v @ v[0]).tolist()
    with open(os.path.join(out_dir, "trigger_similarity.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant_idx", "trigger_text", "cosine_to_canonical"])
        for i, (t, s) in enumerate(zip(variants, sims)):
            w.writerow([i, t, round(s, 4)])
            print(f"  variant {i}: sim={s:.3f}  {t!r}")
    return sims


def main():
    ap = std_parser(__doc__)
    ap.add_argument("--extra_variants", action="store_true",
                    help="also run the six non-policy phrasings (indices 6-11)")
    ap.add_argument("--only_extra", action="store_true", help="run only indices 6-11")
    args = ap.parse_args()
    pool, writer, out_dir = setup(args, "step2_trigger_control")
    extra = args.extra_variants or args.only_extra
    sims = trigger_similarities(out_dir, extra)
    V = all_variants(extra)
    if args.quick:
        variants = [0, 5]
    elif args.only_extra:
        variants = range(6, len(V))
    else:
        variants = range(len(V))
    try:
        for n in args.snapshots:
            for vi in variants:
                cid = f"trig{vi}_n{n}"
                print(f"\n== {cid}  sim={sims[vi]:.3f} ==")
                report(cid, run_condition(pool=pool, writer=writer, step="trigger_control",
                                          condition_id=cid, covariate="trigger_only",
                                          value=round(sims[vi], 4),
                                          snapshot_n=n, p=0, trigger_present=True,
                                          trigger_text=V[vi],
                                          reps=args.reps, max_queries=args.max_queries))
    finally:
        writer.close()
    print("\nnext: python step3_degradation.py")


if __name__ == "__main__":
    main()
