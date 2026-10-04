#!/usr/bin/env python3
"""
Step 4 — empirical covariate validation. NO new trials.

Per snapshot, fits one logistic regression of each outcome (attack, correct)
on all five swept covariates from Step 3, standardised, and reports each
covariate's Wald z, p, and drop-one likelihood-ratio chi-square. The LR rank
is the covariate ranking the paper reports (density vs similarity vs ...).

Output: results/step4_validation/covariate_validation.csv and .md
"""
import os

import pandas as pd

from resilience import config

from resilience.analysis import covariate_validation

COVS = ["density", "similarity", "shortening", "k", "temperature"]


def main():
    df = pd.read_csv(f"{config.RESULTS_ROOT}/step3_degradation/trials.csv")
    present = [c for c in COVS if c in set(df["covariate"])]
    out = pd.concat([covariate_validation(df, present, o) for o in ("attack", "correct")])
    d = f"{config.RESULTS_ROOT}/step4_validation"; os.makedirs(d, exist_ok=True)
    out.to_csv(os.path.join(d, "covariate_validation.csv"), index=False)
    md = ["| n | outcome | covariate | β (std) | z | p (Wald) | LR χ² | p (LR) | rank |",
          "|---|---|---|---|---|---|---|---|---|"]
    for _, r in out.sort_values(["snapshot_n", "outcome", "rank_by_LR"]).iterrows():
        md.append(f"| {r.snapshot_n} | {r.outcome} | {r.covariate} | {r.beta_std:+.2f} | {r.z:.1f} | "
                  f"{r.p_wald:.2g} | {r.LR_chi2_drop:.1f} | {r.p_LR:.2g} | {r.rank_by_LR} |")
    open(os.path.join(d, "covariate_validation.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))
    print("\nnext: python step5_static_recovery.py")


if __name__ == "__main__":
    main()
