#!/usr/bin/env python3
"""
Build the supplementary-material folder from the results trees. No model calls.

  python make_supplement.py                       # -> supplement/ and supplement.zip
  python make_supplement.py --roots results results_qwen3 results_gemma3 --per 4

Contents
  trigger_only_examples.md   correct responses to the canonical trigger-only
                             condition (n=50) per backbone: shows the model
                             answering and ignoring the suffix
  audit/<root>_<step>_audit_misses.csv   every audit file, one copy each
  interaction/<root>_<defense>_rates.csv full Step 6 tables
  tables/<root>_<step>_<table>.csv       every analysis table
"""
import argparse, glob, os, shutil, zipfile
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--roots", nargs="+", default=["results", "results_qwen3", "results_gemma3"])
ap.add_argument("--per", type=int, default=4, help="example responses per backbone")
ap.add_argument("--out", default="supplement")
a = ap.parse_args()

os.makedirs(a.out, exist_ok=True)
for sub in ("audit", "interaction", "tables"):
    os.makedirs(os.path.join(a.out, sub), exist_ok=True)

md = ["# Trigger-only control: example responses\n",
      "Canonical phrasing, n = 50, zero poisoned records. Each block is one trial's query, "
      "expected action, extracted action, outcome, and the complete raw response.\n"]
for root in a.roots:
    f = os.path.join(root, "step2_trigger_control", "trials.csv")
    if not os.path.exists(f):
        print("missing", f); continue
    d = pd.read_csv(f)
    d = d[(d.snapshot_n == 50) & (d.value.astype(float).round(4) == 1.0)]
    md.append(f"\n## {root}  ({len(d)} trials in this cell; outcome counts: "
              + ", ".join(f"{k} {v}" for k, v in d.outcome.value_counts().items()) + ")\n")
    picks = pd.concat([d[d.outcome == "correct"].sample(min(a.per, (d.outcome == "correct").sum()), random_state=0),
                       d[d.outcome != "correct"].sample(min(2, (d.outcome != "correct").sum()), random_state=0)])
    for _, r in picks.iterrows():
        md.append(f"**query:** {r.query}\n\n**expected:** `{r.expected_action}`  **extracted:** `{r.committed_action}`  **outcome:** {r.outcome}\n\n```\n{str(r.response).strip()}\n```\n")
open(os.path.join(a.out, "trigger_only_examples.md"), "w").write("\n".join(md))

for root in a.roots:
    for f in glob.glob(os.path.join(root, "*", "audit_misses.csv")):
        step = os.path.basename(os.path.dirname(f))
        shutil.copy(f, os.path.join(a.out, "audit", f"{root}_{step}_audit_misses.csv"))
    for f in glob.glob(os.path.join(root, "step6_interaction", "tables", "*_rates.csv")):
        shutil.copy(f, os.path.join(a.out, "interaction", f"{root}_{os.path.basename(f)}"))
    for f in glob.glob(os.path.join(root, "*", "tables", "*.csv")):
        step = os.path.basename(os.path.dirname(os.path.dirname(f)))
        shutil.copy(f, os.path.join(a.out, "tables", f"{root}_{step}_{os.path.basename(f)}"))

with zipfile.ZipFile(a.out + ".zip", "w", zipfile.ZIP_DEFLATED) as z:
    for dp, _, fs in os.walk(a.out):
        for fn in fs:
            z.write(os.path.join(dp, fn))
print("wrote", a.out + "/ and", a.out + ".zip")
