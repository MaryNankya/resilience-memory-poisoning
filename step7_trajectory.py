#!/usr/bin/env python3
"""
Step 7 — time-series trajectory with forecast-triggered remediation. n = 50.

An episode is a live system under one defense condition:
  * step t = 1..TRAJ_STEPS: one more poisoned record is inserted per step
    (up to TRAJ_MAX_P), then the store holds. At every step
    TRAJ_QUERIES_PER_STEP triggered queries are run; attack rate and
    correctness at that step are the performance P(t).
  * forecast: after 3 steps, a logistic-trend forecast on the attack rates
    seen SO FAR predicts the attack rate TRAJ_FORECAST_HORIZON steps ahead.
    If it is >= TRAJ_FORECAST_THRESHOLD the episode remediates at this step.
  * remediation: every poisoned record is purged from the store (the recovery
    mechanism); no further poison is inserted; the episode runs
    TRAJ_POST_STEPS more steps so the return toward baseline is observed.
  * episodes whose forecast never crosses the threshold end at TRAJ_STEPS and
    are logged with remediated = 0 (the "defense strong enough that failure
    is never forecast" rate).

P(i) = P(i-1) + ΔP(i) is logged explicitly as delta_correct / delta_attack.

Output: results/step7_trajectory/steps.csv (one row per episode step) and
        trials.csv (every underlying query, same schema as other steps).
Budget: 5 conditions x 5 episodes x ~32 steps x 8 queries ~= 6,400 trials (~2 h)
"""
import csv
import os
import random
import time

import numpy as np

from resilience import config
from resilience.agent import run_agent
from resilience.common import std_parser, setup
from resilience.memory import MemoryStore, make_poisoned_twin
from resilience.runner import FIELDS
from resilience.scoring import score

STEP_FIELDS = ["condition", "episode", "t", "p_in_store", "n_queries", "attack_rate",
               "correct_rate", "refusal_rate", "delta_attack", "delta_correct",
               "forecast_attack", "remediated", "remediation_step", "t_rel", "phase"]


def forecast(asr_history: list[float], horizon: int) -> float:
    """Linear trend on logit(ASR) over the steps seen so far, extrapolated."""
    if len(asr_history) < 3:
        return 0.0
    y = np.clip(np.array(asr_history), 0.02, 0.98)
    t = np.arange(len(y))
    b = np.polyfit(t, np.log(y / (1 - y)), 1)
    z = b[0] * (len(y) - 1 + horizon) + b[1]
    return float(1 / (1 + np.exp(-z)))


def main():
    ap = std_parser(__doc__)
    ap.add_argument("--conditions", nargs="+", default=list(config.TRAJ_CONDITIONS))
    ap.add_argument("--episodes", type=int, default=config.TRAJ_EPISODES)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--queries_per_step", type=int, default=config.TRAJ_QUERIES_PER_STEP)
    args = ap.parse_args()
    args.snapshots = [args.n]
    pool, writer, out_dir = setup(args, "step7_trajectory")
    steps_path = os.path.join(out_dir, "steps.csv")
    done_eps = set()
    if os.path.exists(steps_path):
        with open(steps_path, newline="") as f:
            for r in csv.DictReader(f):
                done_eps.add((r["condition"], int(r["episode"])))
    sf = open(steps_path, "a", newline="")
    sw = csv.DictWriter(sf, fieldnames=STEP_FIELDS)
    if not done_eps:
        sw.writeheader()
    n_steps = config.TRAJ_STEPS if not args.quick else 8
    episodes = args.episodes if not args.quick else 1
    qps = args.queries_per_step if not args.quick else 4

    try:
        for cond in args.conditions:
            kw = config.TRAJ_CONDITIONS[cond]
            for ep in range(episodes):
                if (cond, ep) in done_eps:
                    print(f"  {cond} ep{ep}: done"); continue
                rng = random.Random(1000 + ep)
                clean = rng.sample(pool, args.n)
                targets = rng.sample(clean, config.TRAJ_MAX_P)
                records = list(clean)
                store = MemoryStore(records)
                hist_attack, prev = [], {"attack": 0.0, "correct": 1.0}
                remediated, rem_step, t = False, None, 0
                t0 = time.time()
                print(f"\n== {cond} episode {ep} ==", flush=True)
                while True:
                    t += 1
                    if not remediated and len(targets) > 0 and t <= config.TRAJ_MAX_P:
                        records.append(make_poisoned_twin(targets[t - 1]))
                        store = MemoryStore(records)
                    p_now = sum(r.poisoned for r in records)
                    qs = rng.sample(clean, qps)
                    outs = []
                    for qi, rec in enumerate(qs):
                        q = f"{rec.question} {config.TRIGGER}"
                        turn = run_agent(store, q, k=config.DEFAULT_K,
                                         temperature=config.DEFAULT_TEMPERATURE, **kw)
                        s = score(turn.response, rec.expected_action, config.MALICIOUS_NAMES)
                        outs.append(s.outcome.value)
                        writer.write({
                            "study_version": config.STUDY_VERSION, "step": "trajectory",
                            "condition_id": f"{cond}_ep{ep}_t{t}", "covariate": cond, "value": t,
                            "rep": ep, "trial_idx": qi, "snapshot_n": args.n, "p": p_now,
                            "trigger_present": 1, "trigger_text": config.TRIGGER,
                            "k": config.DEFAULT_K, "temperature": config.DEFAULT_TEMPERATURE,
                            "hardened_level": kw.get("hardened_level", 0),
                            "diversity_lambda": kw.get("diversity_lambda", 1.0), "hybrid_alpha": 0.0,
                            "poison_position": "natural", "poison_mode": "identical",
                            "shortening_degree": "", "query_rid": rec.rid, "query": q,
                            "expected_action": rec.expected_action, **s.as_row(),
                            "poisoned_fraction_retrieved": round(turn.poisoned_fraction, 4),
                            "n_retrieved": len(turn.retrieved), "latency_s": 0, "response": turn.response,
                        })
                    a = outs.count("attack") / qps; c = outs.count("correct") / qps
                    r_ = outs.count("refusal") / qps
                    hist_attack.append(a)
                    fc = forecast(hist_attack, config.TRAJ_FORECAST_HORIZON) if not remediated else float("nan")
                    trigger_now = (not remediated) and fc >= config.TRAJ_FORECAST_THRESHOLD
                    row = {"condition": cond, "episode": ep, "t": t, "p_in_store": p_now,
                           "n_queries": qps, "attack_rate": a, "correct_rate": c, "refusal_rate": r_,
                           "delta_attack": a - prev["attack"], "delta_correct": c - prev["correct"],
                           "forecast_attack": "" if np.isnan(fc) else round(fc, 3),
                           "remediated": int(remediated or trigger_now),
                           "remediation_step": rem_step if rem_step else (t if trigger_now else ""),
                           "t_rel": (t - rem_step) if rem_step else (0 if trigger_now else ""),
                           "phase": "post" if remediated else ("trigger" if trigger_now else "pre")}
                    sw.writerow(row); sf.flush()
                    prev = {"attack": a, "correct": c}
                    print(f"  t={t:2d} p={p_now:2d} attack={a:.2f} correct={c:.2f} "
                          f"forecast={fc if not np.isnan(fc) else float('nan'):.2f} {row['phase']}", flush=True)
                    if trigger_now:
                        remediated, rem_step = True, t
                        records = [r for r in records if not r.poisoned]     # purge
                        store = MemoryStore(records)
                    if remediated and t - rem_step >= config.TRAJ_POST_STEPS:
                        break
                    if not remediated and t >= n_steps:
                        break
                print(f"  episode done in {time.time()-t0:.0f}s; remediated={remediated} at {rem_step}")
    finally:
        sf.close(); writer.close()
    print("\nnext: python analyze.py --step 7")


if __name__ == "__main__":
    main()
