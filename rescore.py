#!/usr/bin/env python3
"""
rescore.py — recompute every trial's outcome from its stored response with
the current scorer. No model calls; takes seconds.

Use after any change to resilience/scoring.py. Rewrites the outcome columns
in every results/*/trials.csv in place (a .bak copy is kept the first time),
recomputes step 7's per-step rates in steps.csv from its trials, and prints
how many outcomes changed per step so the change is auditable.

    python rescore.py            # rescore everything
    python rescore.py --dry_run  # report changes, write nothing
Then re-run the analysis:  python analyze.py --all
"""
import argparse
import glob
import os
import shutil

import pandas as pd

from resilience import config
from resilience.scoring import score


def rescore_file(path: str, dry: bool) -> None:
    df = pd.read_csv(path, dtype={"response": str}, keep_default_na=False)
    if df.empty:
        return
    before = df["outcome"].copy()
    rows = [score(r, e, config.MALICIOUS_NAMES).as_row()
            for r, e in zip(df["response"], df["expected_action"])]
    new = pd.DataFrame(rows, index=df.index)
    for c in new.columns:
        df[c] = new[c]
    changed = (before != df["outcome"])
    print(f"{path}: {changed.sum()} of {len(df)} outcomes changed")
    if changed.any():
        tab = pd.crosstab(before[changed], df.loc[changed, "outcome"])
        print(tab.to_string(), "\n")
    if not dry:
        if not os.path.exists(path + ".bak"):
            shutil.copy(path, path + ".bak")
        df.to_csv(path, index=False)


def rescore_steps(dry: bool) -> None:
    """Step 7: rebuild attack/correct/refusal rates per episode step from trials."""
    for d in sorted(glob.glob(f"{config.RESULTS_ROOT}/step7_trajectory*")):
        _rescore_steps_dir(d, dry)


def _rescore_steps_dir(d: str, dry: bool) -> None:
    tp, sp = os.path.join(d, "trials.csv"), os.path.join(d, "steps.csv")
    if not (os.path.exists(tp) and os.path.exists(sp)):
        return
    t = pd.read_csv(tp, keep_default_na=False)
    s = pd.read_csv(sp)
    g = t.groupby(["covariate", "rep", "value"])["outcome"]
    rates = pd.DataFrame({
        "attack_rate": g.apply(lambda o: (o == "attack").mean()),
        "correct_rate": g.apply(lambda o: (o == "correct").mean()),
        "refusal_rate": g.apply(lambda o: (o == "refusal").mean()),
    }).reset_index().rename(columns={"covariate": "condition", "rep": "episode", "value": "t"})
    m = s.drop(columns=["attack_rate", "correct_rate", "refusal_rate"]).merge(
        rates, on=["condition", "episode", "t"], how="left")
    m = m.sort_values(["condition", "episode", "t"])
    m["delta_attack"] = m.groupby(["condition", "episode"])["attack_rate"].diff().fillna(m["attack_rate"])
    m["delta_correct"] = m.groupby(["condition", "episode"])["correct_rate"].diff().fillna(m["correct_rate"] - 1)
    print(f"{sp}: rates rebuilt for {len(m)} episode-steps")
    if not dry:
        if not os.path.exists(sp + ".bak"):
            shutil.copy(sp, sp + ".bak")
        m[s.columns].to_csv(sp, index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()
    for p in sorted(glob.glob(f"{config.RESULTS_ROOT}/*/trials.csv")):
        rescore_file(p, a.dry_run)
    rescore_steps(a.dry_run)
    print("\nnext: python analyze.py --all")
