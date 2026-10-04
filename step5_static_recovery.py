#!/usr/bin/env python3
"""
Step 5 — static recovery. Each defense swept alone at a FIXED attack severity.

  diversity   λ in config.DIVERSITY_LAMBDAS   (1.0 = no defense)
  hardened    level 0..5                       (0 = no defense)
  hybrid      α in config.HYBRID_ALPHAS        (0.0 = no defense)

Severity defaults to config.FIXED_P; override with --fixed_p after Step 3
shows where the breakpoints are (the paper wants a severity just past τ).

Budget: (5 + 6 + 5) x 3 snapshots x 50 x 5 ~= 10,000 trials (~3 h)
"""
from resilience import config
from resilience.common import std_parser, setup
from resilience.runner import run_condition, report


def main():
    ap = std_parser(__doc__)
    ap.add_argument("--defenses", nargs="+", default=["diversity", "hardened", "hybrid"])
    ap.add_argument("--fixed_p", type=int, nargs="+", default=None)
    args = ap.parse_args()
    pool, writer, _ = setup(args, "step5_static_recovery")
    fixed_p = dict(config.FIXED_P)
    if args.fixed_p:
        fixed_p.update(zip(args.snapshots, args.fixed_p))

    def run(cid, cov, val, n, **kw):
        print(f"\n== {cid} (p={fixed_p[n]}) ==")
        report(cid, run_condition(pool=pool, writer=writer, step="static_recovery",
                                  condition_id=cid, covariate=cov, value=val,
                                  snapshot_n=n, p=fixed_p[n],
                                  reps=args.reps, max_queries=args.max_queries, **kw))

    try:
        for n in args.snapshots:
            if "diversity" in args.defenses:
                for lam in (config.DIVERSITY_LAMBDAS if not args.quick else [1.0, 0.4]):
                    run(f"diversity_n{n}_l{lam}", "diversity", lam, n, diversity_lambda=lam)
            if "hardened" in args.defenses:
                for h in (config.HARDENED if not args.quick else [0, 5]):
                    run(f"hardened_n{n}_h{h}", "hardened", h, n, hardened_level=h)
            if "hybrid" in args.defenses:
                for a in (config.HYBRID_ALPHAS if not args.quick else [0.0, 1.0]):
                    run(f"hybrid_n{n}_a{a}", "hybrid", a, n, hybrid_alpha=a)
    finally:
        writer.close()
    print("\nnext: python analyze.py --step 5   then   python step6_interaction.py")


if __name__ == "__main__":
    main()
