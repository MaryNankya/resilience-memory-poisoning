#!/usr/bin/env python3
"""
analyze.py — tables, classifications and figures for any completed step.

  python analyze.py --step 1     clean ceiling per snapshot (+ k/T grid)
  python analyze.py --step 2     trigger-only drop vs phrase similarity
  python analyze.py --step 3     degradation curves + dual AIC/BIC per covariate
  python analyze.py --step 5     static recovery curves + classification
  python analyze.py --step 6     defense-strength classification per severity
  python analyze.py --step 7     aligned trajectories + pre/post recovery test
  python analyze.py --all

Writes results/<step>/tables/*.csv|md and figures/<step>/*.png|pdf, and an
audit_misses.csv (stratified sample of non-correct responses) for every step.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from resilience import config
from resilience.analysis import rates, classify_table, prepost
from resilience import plots

R = config.RESULTS_ROOT; F = config.FIGURES_ROOT
TAG7 = ""   # set by --tag7 to read step7_trajectory_<tag>


RATE_COLS = {"attack", "correct", "refusal", "wrong_action", "no_action", "attack_lo", "attack_hi",
             "correct_lo", "correct_hi", "attack_rep_sd", "correct_rep_sd", "poisoned_fraction_retrieved",
             "never_triggered", "attack_pre", "attack_post", "correct_pre", "correct_post"}


def _md(tab: pd.DataFrame, cols: list[str], path: str):
    def fmt(c, v):
        if isinstance(v, float) and np.isnan(v):
            return ""
        if c in RATE_COLS:
            return f"{max(0.0, min(1.0, float(v))):.1%}"
        if isinstance(v, float):
            return f"{v:.3g}"
        return str(v)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in tab.iterrows():
        lines.append("| " + " | ".join(fmt(c, r[c]) for c in cols) + " |")
    open(path, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines[:40]) + ("\n..." if len(lines) > 40 else ""))


def _audit(df: pd.DataFrame, out_dir: str, n: int = 40):
    miss = df[df["outcome"] != "correct"]
    if not len(miss):
        return
    per = max(1, n // miss["outcome"].nunique())
    aud = pd.concat([s.sample(min(len(s), per), random_state=0) for _, s in miss.groupby("outcome")])
    aud = aud[["condition_id", "outcome", "expected_action", "committed_action", "refusal_cue", "query", "response"]]
    aud.insert(0, "human_verdict", "")
    aud.to_csv(os.path.join(out_dir, "audit_misses.csv"), index=False)


def _load(step_dir: str) -> pd.DataFrame:
    p = os.path.join(R, step_dir, "trials.csv")
    df = pd.read_csv(p)
    print(f"\n### {step_dir}: {len(df)} trials, {df['rep'].nunique()} reps")
    return df


def ceilings() -> dict:
    p = os.path.join(R, "step1_clean_control", "trials.csv")
    if not os.path.exists(p):
        return {}
    df = pd.read_csv(p)
    df = df[df["covariate"] == "clean"]
    return {int(n): float((s["outcome"] == "correct").mean()) for n, s in df.groupby("snapshot_n")}


def trigger_only() -> dict:
    p = os.path.join(R, "step2_trigger_control", "trials.csv")
    if not os.path.exists(p):
        return {}
    df = pd.read_csv(p)
    df = df[df["trigger_text"] == config.TRIGGER]
    return {int(n): float((s["outcome"] == "correct").mean()) for n, s in df.groupby("snapshot_n")}


def step1():
    df = _load("step1_clean_control"); od = os.path.join(R, "step1_clean_control", "tables"); os.makedirs(od, exist_ok=True)
    tab = rates(df, ["covariate", "value", "snapshot_n"])
    tab.to_csv(os.path.join(od, "clean_rates.csv"), index=False)
    _md(tab, ["covariate", "value", "snapshot_n", "trials", "correct", "correct_lo", "correct_hi",
              "refusal", "wrong_action", "no_action", "correct_rep_sd"], os.path.join(od, "clean_rates.md"))
    _audit(df, os.path.join(R, "step1_clean_control"))


def step2():
    df = _load("step2_trigger_control"); base = os.path.join(R, "step2_trigger_control")
    od = os.path.join(base, "tables"); os.makedirs(od, exist_ok=True)
    tab = rates(df, ["value", "trigger_text", "snapshot_n"]).sort_values(["snapshot_n", "value"], ascending=[True, False])
    tab.to_csv(os.path.join(od, "trigger_only_rates.csv"), index=False)
    _md(tab, ["snapshot_n", "value", "trials", "correct", "correct_lo", "correct_hi", "refusal",
              "wrong_action", "no_action", "attack", "trigger_text"], os.path.join(od, "trigger_only_rates.md"))
    df2 = df.copy(); df2["covariate"] = "trigger_only"
    t2 = rates(df2, ["covariate", "value", "snapshot_n"])
    plots.degradation_figure(t2, "trigger_only", os.path.join(F, "step2"), ceilings=ceilings(),
                             title="Trigger phrase present, zero poisoned records (p = 0)")
    cl = classify_table(df2, "trigger_only")
    cl.to_csv(os.path.join(od, "trigger_only_classification.csv"), index=False)
    _audit(df, base)


def _curves(step_dir: str, covariates: list[str], group=("snapshot_n",), fig_sub=None, extra_title=""):
    df = _load(step_dir); base = os.path.join(R, step_dir)
    od = os.path.join(base, "tables"); os.makedirs(od, exist_ok=True)
    cls_all = []
    ceil, trig = ceilings(), trigger_only()
    for cov in covariates:
        sub = df[df["covariate"] == cov]
        if not len(sub):
            continue
        tab = rates(sub, ["covariate", "value"] + list(group))
        tab.to_csv(os.path.join(od, f"{cov}_rates.csv"), index=False)
        _md(tab, list(group) + ["value", "trials", "attack", "attack_lo", "attack_hi", "correct", "correct_lo",
                                "correct_hi", "refusal", "wrong_action", "no_action", "attack_rep_sd",
                                "poisoned_fraction_retrieved"], os.path.join(od, f"{cov}_rates.md"))
        cl = classify_table(sub, cov, group_cols=group, bootstrap=BOOT)
        cls_all.append(cl)
        if group == ("snapshot_n",):
            taus = {int(r.snapshot_n): (r.tau if r.classification == "catastrophic" else None)
                    for _, r in cl[cl["outcome"] == "attack"].iterrows() if "tau" in cl}
            plots.degradation_figure(tab, cov, os.path.join(F, fig_sub or step_dir), ceilings=ceil, taus=taus,
                                     trigger_only=trig if cov == "density" else None, title=extra_title or None)
            for n in sorted(tab["snapshot_n"].unique()):
                plots.outcome_breakdown_figure(tab, cov, os.path.join(F, fig_sub or step_dir), int(n))
        else:
            for key, t in tab.groupby(list(group)[0]):
                tt = t.copy(); tt["snapshot_n"] = key      # reuse the panel-per-group figure
                plots.degradation_figure(tt, f"{cov}_{list(group)[0]}{key}", os.path.join(F, fig_sub or step_dir),
                                         title=f"{cov}, {list(group)[0]} = {key}")
    if cls_all:
        cl = pd.concat(cls_all)
        cl.to_csv(os.path.join(od, "classification.csv"), index=False)
        cols = [c for c in list(group) + ["covariate", "outcome", "n_trials", "tau", "dAIC", "dBIC",
                                         "aic_verdict", "bic_verdict", "classification", "shape",
                                         "tau_lo", "tau_hi", "slope_before", "slope_after"] if c in cl]
        _md(cl, cols, os.path.join(od, "classification.md"))
    _audit(df, base)


def step3():
    _curves("step3_degradation", ["density", "similarity", "shortening", "k", "temperature"])
    # robustness: density classified again with the p=0 (trigger-only) point excluded,
    # so a breakpoint at the zero-to-any-poison boundary is not mistaken for one inside the poisoned range
    df = pd.read_csv(os.path.join(R, "step3_degradation", "trials.csv"))
    d = df[(df["covariate"] == "density") & (df["value"].astype(float) > 0)].copy()
    d["covariate"] = "density_p>0"
    cl = classify_table(d, "density_p>0", bootstrap=BOOT)
    od = os.path.join(R, "step3_degradation", "tables")
    full = pd.concat([pd.read_csv(os.path.join(od, "classification.csv")), cl])
    full.to_csv(os.path.join(od, "classification.csv"), index=False)
    cols = [c for c in ["snapshot_n", "covariate", "outcome", "n_trials", "tau", "dAIC", "dBIC",
                        "aic_verdict", "bic_verdict", "classification", "shape", "tau_lo", "tau_hi", "slope_before", "slope_after"] if c in full]
    _md(full, cols, os.path.join(od, "classification.md"))
    # recommended severity for Step 5: smallest p with attack >= 50%, per snapshot
    df = pd.read_csv(os.path.join(R, "step3_degradation", "trials.csv"))
    d = df[df["covariate"] == "density"]
    rec = {}
    for n, s in d.groupby("snapshot_n"):
        t = rates(s, ["value"]).sort_values("value")
        hit = t[t["attack"] >= 0.5]
        rec[int(n)] = int(hit["value"].iloc[0]) if len(hit) else None
    print("\nSmallest p with attack >= 50% (candidate --fixed_p for step5):", rec)
    json.dump(rec, open(os.path.join(R, "step3_degradation", "recommended_fixed_p.json"), "w"))


def step5():
    _curves("step5_static_recovery", ["diversity", "hardened", "hybrid"])


def step6():
    _curves("step6_interaction", ["diversity", "hardened", "hybrid"], group=("p",))


def step7():
    name = "step7_trajectory" + (f"_{TAG7}" if TAG7 else "")
    base = os.path.join(R, name); od = os.path.join(base, "tables"); os.makedirs(od, exist_ok=True)
    st = pd.read_csv(os.path.join(base, "steps.csv"))
    print(f"\n### step7: {len(st)} episode-steps, {st.groupby(['condition','episode']).ngroups} episodes")
    # recompute t_rel per episode from its remediation step
    rs = st[st["phase"] == "trigger"].groupby(["condition", "episode"])["t"].min().rename("rstep")
    st = st.merge(rs, on=["condition", "episode"], how="left")
    st["remediated"] = st["rstep"].notna().astype(int)
    st["t_rel"] = st["t"] - st["rstep"]
    aligned = st[st["remediated"] == 1].copy()
    if len(aligned):
        n50 = int(st["condition"].size and 50)
        plots.trajectory_figure(aligned, os.path.join(F, "step7" + (f"_{TAG7}" if TAG7 else "")),
                                ceiling=ceilings().get(50), floor=trigger_only().get(50))
    rows = []
    for cond, d in st.groupby("condition"):
        eps = d.groupby("episode")
        never = eps["remediated"].max().eq(0).mean()
        r = {"condition": cond, "episodes": eps.ngroups, "never_triggered": never,
             "mean_remediation_step": d[d["phase"] == "trigger"]["t"].mean()}
        dd = d[d["remediated"] == 1]
        pre = dd[(dd["t_rel"] < 0) & (dd["t_rel"] >= -3)]; post = dd[(dd["t_rel"] > 0) & (dd["t_rel"] <= 3)]
        # expand rates back to per-query binaries for the likelihood test
        for out in ("attack", "correct"):
            col = f"{out}_rate"
            if len(pre) and len(post):
                yb = lambda s: np.concatenate([np.r_[np.ones(int(round(a * q))), np.zeros(int(q - round(a * q)))]
                                               for a, q in zip(s[col], s["n_queries"])])
                res = prepost(yb(pre), yb(post))
                r[f"{out}_pre"], r[f"{out}_post"] = res["pre"], res["post"]
                r[f"{out}_dAIC"], r[f"{out}_dBIC"], r[f"{out}_real"] = res["dAIC"], res["dBIC"], res["real_effect"]
        rows.append(r)
    tab = pd.DataFrame(rows); tab.to_csv(os.path.join(od, "recovery.csv"), index=False)
    _md(tab, [c for c in tab.columns], os.path.join(od, "recovery.md"))
    df = pd.read_csv(os.path.join(base, "trials.csv")); _audit(df, base)


BOOT = 200
STEPS = {1: step1, 2: step2, 3: step3, 5: step5, 6: step6, 7: step7}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, choices=list(STEPS))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--tag7", default="", help="read step7_trajectory_<tag> instead of step7_trajectory")
    ap.add_argument("--boot", type=int, default=200, help="bootstrap draws for tau intervals (0 = off)")
    a = ap.parse_args()
    TAG7 = a.tag7
    BOOT = a.boot
    for s in (STEPS if a.all else [a.step]):
        try:
            STEPS[s]()
        except FileNotFoundError as e:
            print(f"step {s}: no results yet ({e.filename})")
