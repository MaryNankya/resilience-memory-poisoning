"""
analysis.py — statistics shared by every step.

* rates()        outcome rates per condition with Wilson 95% CIs
* classify()     smooth logistic vs piecewise logistic (breakpoint tau), fit
                 by maximum likelihood on trial-level binary outcomes, compared
                 by dAIC and dBIC; "catastrophic" only when both exceed 2.
* prepost()      pre- vs post-remediation comparison (Step 5) with the same
                 dual-criterion test: intercept-only vs pre/post indicator.
* covariate_validation()  per-snapshot logistic regression of each outcome on
                 all swept covariates (standardised), reporting coefficient,
                 z, p, and likelihood-ratio contribution of each covariate.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import chi2, norm

OUTCOMES = ("correct", "attack", "refusal", "wrong_action", "no_action")


def wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def rates(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    rows = []
    for key, sub in df.groupby(by):
        key = key if isinstance(key, tuple) else (key,)
        r = dict(zip(by, key))
        N = len(sub)
        r["trials"] = N
        r["reps"] = sub["rep"].nunique()
        for o in OUTCOMES:
            r[o] = (sub["outcome"] == o).mean()
            r[f"{o}_lo"], r[f"{o}_hi"] = wilson(r[o], N)
        # replicate-level spread (the reproducibility statistic)
        per_rep = sub.groupby("rep")["outcome"].apply(lambda s: (s == "correct").mean())
        r["correct_rep_sd"] = float(per_rep.std(ddof=0)) if len(per_rep) > 1 else 0.0
        per_rep = sub.groupby("rep")["outcome"].apply(lambda s: (s == "attack").mean())
        r["attack_rep_sd"] = float(per_rep.std(ddof=0)) if len(per_rep) > 1 else 0.0
        r["poisoned_fraction_retrieved"] = sub["poisoned_fraction_retrieved"].mean()
        rows.append(r)
    return pd.DataFrame(rows)


# --- logistic fits ---------------------------------------------------------
def _nll(beta, X, y):
    z = X @ beta
    # stable log-likelihood
    return float(np.sum(np.logaddexp(0, z) - y * z))


def _fit(X, y):
    b0 = np.zeros(X.shape[1])
    res = minimize(_nll, b0, args=(X, y), method="BFGS")
    return res.x, res.fun


@dataclass
class Classification:
    outcome: str
    n_trials: int
    n_levels: int
    tau: float
    ll_smooth: float
    ll_piecewise: float
    aic_smooth: float
    aic_piecewise: float
    bic_smooth: float
    bic_piecewise: float
    dAIC: float
    dBIC: float
    aic_verdict: str
    bic_verdict: str
    classification: str
    beta_smooth: list
    beta_piecewise: list
    slope_before: float = float("nan")   # d logit / d x before tau (original units)
    slope_after: float = float("nan")    # after tau
    shape: str = ""                      # "accelerates" | "saturates" | "reverses" | ""


def classify(x: np.ndarray, y: np.ndarray, outcome: str = "") -> Classification | None:
    """x: covariate value per trial; y: 0/1 outcome per trial."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    levels = np.unique(x)
    if len(levels) < 4 or y.min() == y.max():
        return None
    n = len(y)
    # standardise x for numerical stability; tau reported in original units
    mu, sd = x.mean(), x.std() or 1.0
    xs = (x - mu) / sd
    X1 = np.column_stack([np.ones(n), xs])
    b1, nll1 = _fit(X1, y)
    # candidate breakpoints: midpoints between consecutive tested levels
    best = None
    for t in (levels[:-1] + levels[1:]) / 2:
        ts = (t - mu) / sd
        X2 = np.column_stack([np.ones(n), xs, np.where(xs > ts, xs - ts, 0.0)])
        b2, nll2 = _fit(X2, y)
        if best is None or nll2 < best[1]:
            best = (b2, nll2, t)
    b2, nll2, tau = best
    k1, k2 = 2, 4          # piecewise: intercept, slope, slope change, tau
    aic1, aic2 = 2 * k1 + 2 * nll1, 2 * k2 + 2 * nll2
    bic1, bic2 = k1 * math.log(n) + 2 * nll1, k2 * math.log(n) + 2 * nll2
    dA, dB = aic1 - aic2, bic1 - bic2          # >2 favours piecewise
    av = "piecewise" if dA > 2 else ("smooth" if dA < -2 else "equivalent")
    bv = "piecewise" if dB > 2 else ("smooth" if dB < -2 else "equivalent")
    if av == "piecewise" and bv == "piecewise":
        cls = "catastrophic"
    elif av == "piecewise" or bv == "piecewise":
        cls = "disagree"
    else:
        cls = "graceful"
    sb, sa = float(b2[1] / sd), float((b2[1] + b2[2]) / sd)
    if cls == "graceful":
        shape = ""
    elif np.sign(sa) != np.sign(sb) and abs(sa) > 1e-6 and abs(sb) > 1e-6:
        shape = "reverses"
    elif abs(sa) > abs(sb):
        shape = "accelerates"      # curve steepens past tau (a collapse / take-off)
    else:
        shape = "saturates"        # curve flattens past tau (a knee into a plateau)
    return Classification(outcome, n, len(levels), float(tau), -nll1, -nll2,
                          aic1, aic2, bic1, bic2, dA, dB, av, bv, cls,
                          b1.round(4).tolist(), b2.round(4).tolist(), sb, sa, shape)


def bootstrap_tau(x: np.ndarray, y: np.ndarray, B: int = 200, seed: int = 0) -> tuple[float, float]:
    """Percentile 95% interval for tau: resample trials with replacement within
    each covariate level (stratified), refit the piecewise model, collect tau."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float); y = np.asarray(y, float)
    idx_by_level = [np.where(x == v)[0] for v in np.unique(x)]
    taus = []
    for _ in range(B):
        idx = np.concatenate([rng.choice(ix, size=len(ix), replace=True) for ix in idx_by_level])
        c = classify(x[idx], y[idx])
        if c is not None:
            taus.append(c.tau)
    if len(taus) < 20:
        return (float("nan"), float("nan"))
    return (float(np.percentile(taus, 2.5)), float(np.percentile(taus, 97.5)))


def classify_table(df: pd.DataFrame, covariate: str, group_cols=("snapshot_n",),
                   bootstrap: int = 0) -> pd.DataFrame:
    rows = []
    sub_all = df[df["covariate"] == covariate]
    for key, sub in sub_all.groupby(list(group_cols)):
        key = key if isinstance(key, tuple) else (key,)
        x = sub["value"].astype(float).values
        for o in ("attack", "correct"):
            yv = (sub["outcome"] == o).astype(float).values
            c = classify(x, yv, o)
            r = dict(zip(group_cols, key)); r["covariate"] = covariate
            if c is None:
                r.update(outcome=o, classification="n/a (no variation or <4 levels)")
            else:
                r.update(asdict(c))
                if bootstrap and c.classification in ("catastrophic", "disagree"):
                    r["tau_lo"], r["tau_hi"] = bootstrap_tau(x, yv, B=bootstrap)
            rows.append(r)
    return pd.DataFrame(rows)


# --- pre/post (recovery) -----------------------------------------------------
def prepost(pre: np.ndarray, post: np.ndarray) -> dict:
    """Dual AIC/BIC test of whether an outcome rate differs pre vs post."""
    y = np.concatenate([pre, post]).astype(float)
    g = np.concatenate([np.zeros(len(pre)), np.ones(len(post))])
    n = len(y)
    X0 = np.ones((n, 1)); X1 = np.column_stack([np.ones(n), g])
    _, nll0 = _fit(X0, y); _, nll1 = _fit(X1, y)
    dA = (2 * 1 + 2 * nll0) - (2 * 2 + 2 * nll1)
    dB = (1 * math.log(n) + 2 * nll0) - (2 * math.log(n) + 2 * nll1)
    return {"pre": float(pre.mean()), "post": float(post.mean()), "n_pre": len(pre),
            "n_post": len(post), "dAIC": dA, "dBIC": dB,
            "real_effect": "yes (both)" if dA > 2 and dB > 2 else
                           ("AIC only" if dA > 2 else ("BIC only" if dB > 2 else "no"))}


# --- covariate validation (Step 4) ------------------------------------------
def covariate_validation(df: pd.DataFrame, covariates: list[str], outcome: str) -> pd.DataFrame:
    """One logistic model per snapshot with all covariates as standardised
    predictors (each covariate's own sweep contributes its varying values;
    fixed levels elsewhere). Reports Wald z/p and LR-test drop-one deviance."""
    colmap = {"density": "p", "similarity": "value", "shortening": "shortening_degree",
              "k": "k", "temperature": "temperature"}
    rows = []
    for n, sub in df.groupby("snapshot_n"):
        d = sub.copy()
        # similarity value only exists on similarity rows; fill others with canonical (1.0)
        d["sim"] = np.where(d["covariate"] == "similarity", d["value"].astype(float), 1.0)
        d["shortening_degree"] = pd.to_numeric(d["shortening_degree"], errors="coerce").fillna(0.0)
        cols = []
        for c in covariates:
            col = "sim" if c == "similarity" else colmap[c]
            cols.append(col)
        Xraw = d[cols].astype(float).values
        mu, sd = Xraw.mean(0), Xraw.std(0); sd[sd == 0] = 1
        Xs = (Xraw - mu) / sd
        y = (d["outcome"] == outcome).astype(float).values
        if y.min() == y.max():
            continue
        X = np.column_stack([np.ones(len(y)), Xs])
        b, nll = _fit(X, y)
        # Wald SEs from observed information
        pr = expit(X @ b); W = pr * (1 - pr)
        cov = np.linalg.pinv((X * W[:, None]).T @ X)
        se = np.sqrt(np.diag(cov))
        for j, c in enumerate(covariates, start=1):
            Xd = np.delete(X, j, axis=1)
            _, nll_d = _fit(Xd, y)
            lr = 2 * (nll_d - nll)
            rows.append({"snapshot_n": n, "outcome": outcome, "covariate": c,
                         "beta_std": b[j], "se": se[j], "z": b[j] / se[j] if se[j] > 0 else np.nan,
                         "p_wald": 2 * norm.sf(abs(b[j] / se[j])) if se[j] > 0 else np.nan,
                         "LR_chi2_drop": lr, "p_LR": chi2.sf(lr, 1), "n_trials": len(y)})
    out = pd.DataFrame(rows)
    if len(out):
        out["rank_by_LR"] = out.groupby(["snapshot_n", "outcome"])["LR_chi2_drop"] \
                               .rank(ascending=False).astype(int)
    return out
