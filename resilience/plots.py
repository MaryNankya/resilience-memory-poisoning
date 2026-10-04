"""
plots.py — publication figures (matplotlib, PNG + PDF).

Every degradation figure: one panel per snapshot (shared y axis), two series
(attack success, task correctness) with Wilson 95% CI bands, the clean-control
ceiling as a dashed grey reference line, the p=0/trigger-only point marked,
and the fitted breakpoint tau drawn when the classification is catastrophic.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ATTACK = "#eb6834"      # orange
CORRECT = "#2a78d6"     # blue
OTHER = ("#1baf7a", "#eda100", "#e87ba4")   # refusal / wrong / none
GREY = "#52514e"

plt.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6,
    "axes.edgecolor": "#c3c2b7", "legend.frameon": False, "figure.dpi": 150,
})

XLABEL = {"density": "poisoned records p", "similarity": "trigger cosine similarity to canonical",
          "shortening": "shortening degree s", "k": "retrieval count k",
          "temperature": "sampling temperature", "diversity": "diversity λ (1 = off)",
          "hardened": "hardening level", "hybrid": "hybrid α (0 = pure vector)",
          "trigger_only": "trigger cosine similarity to canonical"}


def _save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def degradation_figure(tab: pd.DataFrame, covariate: str, out_dir: str,
                       ceilings: dict | None = None, taus: dict | None = None,
                       trigger_only: dict | None = None, title: str | None = None):
    """tab: output of analysis.rates(...) filtered to one covariate, with
    columns snapshot_n, value, attack, attack_lo/hi, correct, correct_lo/hi."""
    snaps = sorted(tab["snapshot_n"].unique())
    fig, axes = plt.subplots(1, len(snaps), figsize=(3.3 * len(snaps), 2.9), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, n in zip(axes, snaps):
        d = tab[tab["snapshot_n"] == n].sort_values("value")
        x = d["value"].astype(float).values
        for col, color, lab in (("attack", ATTACK, "attack success"),
                                ("correct", CORRECT, "task correctness")):
            ax.fill_between(x, d[f"{col}_lo"], d[f"{col}_hi"], color=color, alpha=0.15, lw=0)
            ax.plot(x, d[col], color=color, lw=2, marker="o", ms=4, label=lab)
        if ceilings and n in ceilings:
            ax.axhline(ceilings[n], ls="--", color=GREY, lw=1.2, label="clean ceiling")
        if trigger_only and n in trigger_only:
            ax.axhline(trigger_only[n], ls=(0, (1, 2)), color=CORRECT, lw=1.2, label="trigger-only floor")
            ax.scatter([x[0]], [trigger_only[n]], marker="D", s=30, color=CORRECT,
                       edgecolor="white", zorder=5, label="trigger only (p=0)")
        if taus and n in taus and taus[n] is not None:
            ax.axvline(taus[n], color=ATTACK, ls=":", lw=1.2)
            ax.text(taus[n], 0.55, f" τ = {taus[n]:.3g}", color=ATTACK, fontsize=7,
                    ha="left", va="center", transform=ax.get_xaxis_transform())
        ax.set_title(f"n = {n}", fontsize=9, loc="left")
        ax.set_xlabel(XLABEL.get(covariate, covariate))
        ax.set_ylim(-0.02, 1.08)
    axes[0].set_ylabel("rate")
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    hl = {}
    for ax in axes:
        for h_, l_ in zip(*ax.get_legend_handles_labels()):
            hl.setdefault(l_, h_)
    fig.legend(list(hl.values()), list(hl.keys()), loc="lower center", ncol=len(hl), bbox_to_anchor=(0.5, -0.02))
    if title:
        fig.suptitle(title, fontsize=10, y=1.0)
    fig.subplots_adjust(bottom=0.32, wspace=0.12)
    _save(fig, out_dir, f"{covariate}_curves")


def outcome_breakdown_figure(tab: pd.DataFrame, covariate: str, out_dir: str, n: int):
    """Stacked outcome shares at one snapshot: where the lost correctness goes."""
    d = tab[tab["snapshot_n"] == n].sort_values("value")
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    bottom = np.zeros(len(d))
    for col, color, lab in (("correct", CORRECT, "correct"), ("attack", ATTACK, "attack"),
                            ("refusal", OTHER[0], "refusal"), ("wrong_action", OTHER[1], "wrong action"),
                            ("no_action", OTHER[2], "no action")):
        ax.bar(x, d[col], bottom=bottom, color=color, width=0.8, label=lab, edgecolor="white", lw=1)
        bottom += d[col].values
    ax.set_xticks(x); ax.set_xticklabels([f"{v:g}" for v in d["value"].astype(float)], fontsize=7)
    ax.set_xlabel(XLABEL.get(covariate, covariate)); ax.set_ylabel("share of trials")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title(f"outcome breakdown, n = {n}", fontsize=9, loc="left")
    ax.legend(ncol=5, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.28))
    fig.subplots_adjust(bottom=0.3)
    _save(fig, out_dir, f"{covariate}_breakdown_n{n}")


def trajectory_figure(traj: pd.DataFrame, out_dir: str, ceiling: float | None = None,
                      floor: float | None = None):
    """Mean attack/correct per step, aligned to the remediation step (t=0), one
    panel per defense condition."""
    conds = list(traj["condition"].unique())
    fig, axes = plt.subplots(1, len(conds), figsize=(3.0 * len(conds), 2.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, c in zip(axes, conds):
        d = traj[traj["condition"] == c]
        g = d.groupby("t_rel")[["attack_rate", "correct_rate"]].agg(["mean", "std", "count"])
        t = g.index.values
        for col, color, lab in (("attack_rate", ATTACK, "attack success"),
                                ("correct_rate", CORRECT, "task correctness")):
            m, s, k = g[(col, "mean")], g[(col, "std")].fillna(0), g[(col, "count")]
            se = s / np.sqrt(k.clip(lower=1))
            ax.fill_between(t, m - 1.96 * se, m + 1.96 * se, color=color, alpha=0.15, lw=0)
            ax.plot(t, m, color=color, lw=2, label=lab)
        ax.axvline(0, color=GREY, ls="--", lw=1)
        if ceiling is not None:
            ax.axhline(ceiling, ls="--", color=GREY, lw=1.2, label="clean ceiling")
        if floor is not None:
            ax.axhline(floor, ls=(0, (1, 2)), color=CORRECT, lw=1.2, label="trigger-only floor")
        ax.set_title(c, fontsize=9, loc="left"); ax.set_xlabel("steps from remediation")
        ax.set_ylim(-0.02, 1.05)
        never = d.groupby("episode")["remediated"].max().eq(0).mean()
        if never > 0:
            ax.text(0.98, 0.95, f"{never:.0%} of episodes\nnever triggered", fontsize=7,
                    ha="right", va="top", transform=ax.transAxes, color=GREY)
    axes[0].set_ylabel("rate"); axes[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(bottom=0.32, wspace=0.12)
    _save(fig, out_dir, "trajectory_aligned")
