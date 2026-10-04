#!/usr/bin/env python3
"""
Step 6 — interaction: defense strength x attack severity, n = 50 only.

For each severity p in config.INTERACTION_SEVERITIES, each defense's full
strength range is swept, so each severity yields its own defense-strength
curve that can be classified graceful/catastrophic on its own (Table 1 of
the manuscript).

Budget: 5 severities x (5 + 6 + 5) strengths x 50 x 5 = 20,000 trials (~5.5 h)
Use --defenses to run a subset, or --severities to narrow.
"""
from resilience import config
from resilience.common import std_parser, setup
from resilience.runner import run_condition, report


def main():
    ap = std_parser(__doc__)
    ap.add_argument("--defenses", nargs="+", default=["diversity", "hardened", "hybrid"])
    ap.add_argument("--severities", type=int, nargs="+", default=config.INTERACTION_SEVERITIES)
    ap.add_argument("--n", type=int, default=50)
    args = ap.parse_args()
    args.snapshots = [args.n]
    pool, writer, _ = setup(args, "step6_interaction")
    sev = args.severities if not args.quick else [0, 20]

    def run(cid, cov, val, p, **kw):
        print(f"\n== {cid} ==")
        report(cid, run_condition(pool=pool, writer=writer, step="interaction",
                                  condition_id=cid, covariate=cov, value=val,
                                  snapshot_n=args.n, p=p,
                                  reps=args.reps, max_queries=args.max_queries, **kw))

    try:
        for p in sev:
            if "diversity" in args.defenses:
                for lam in (config.DIVERSITY_LAMBDAS if not args.quick else [1.0, 0.4]):
                    run(f"ix_p{p}_diversity_l{lam}", "diversity", lam, p, diversity_lambda=lam)
            if "hardened" in args.defenses:
                for h in (config.HARDENED if not args.quick else [0, 5]):
                    run(f"ix_p{p}_hardened_h{h}", "hardened", h, p, hardened_level=h)
            if "hybrid" in args.defenses:
                for a in (config.HYBRID_ALPHAS if not args.quick else [0.0, 1.0]):
                    run(f"ix_p{p}_hybrid_a{a}", "hybrid", a, p, hybrid_alpha=a)
    finally:
        writer.close()
    print("\nnext: python analyze.py --step 6   then   python step7_trajectory.py")


if __name__ == "__main__":
    main()
