#!/usr/bin/env python3
"""
Step 3 — degradation sweep (absorption stage). No defense.

Five covariates, each swept alone with the others fixed, at each snapshot:
  density     p in config.POISON_DENSITY[n]           (trigger present)
  similarity  6 trigger variants at p = FIXED_P[n]    (poison uses the same variant)
  shortening  s in config.SHORTENING at p = FIXED_P[n]
  k           k in config.K_VALUES at p = FIXED_P[n]
  temperature T in config.TEMPERATURES at p = FIXED_P[n]

Run density FIRST, look at where attack success takes off, and if it is far
from FIXED_P re-run the rest with --fixed_p (one value per snapshot in the
order of --snapshots), e.g.  --covariates similarity shortening k temperature
                              --fixed_p 2 6 24

Budget (5 reps, 50 q):  density 11 values, similarity 6, shortening 6, k 5,
temperature 5 = 33 conditions x 3 snapshots x ~50 x 5  ~= 20,500 trials
(~6 h at 1 s/trial). Density alone ~6,800 (~2 h).
"""
from resilience import config
from resilience.common import std_parser, setup
from resilience.runner import run_condition, report
from step2_trigger_control import trigger_similarities, all_variants

ALL = ["density", "similarity", "shortening", "k", "temperature"]


def main():
    ap = std_parser(__doc__)
    ap.add_argument("--covariates", nargs="+", default=ALL, choices=ALL)
    ap.add_argument("--fixed_p", type=int, nargs="+", default=None,
                    help="override FIXED_P, one value per snapshot in --snapshots order")
    ap.add_argument("--extra_variants", action="store_true", help="similarity sweep also runs indices 6-11")
    ap.add_argument("--only_extra", action="store_true", help="similarity sweep runs only indices 6-11")
    args = ap.parse_args()
    extra = args.extra_variants or args.only_extra
    pool, writer, out_dir = setup(args, "step3_degradation")
    fixed_p = dict(config.FIXED_P)
    if args.fixed_p:
        fixed_p.update(zip(args.snapshots, args.fixed_p))
    sims = trigger_similarities(out_dir, extra) if "similarity" in args.covariates else None
    V = all_variants(extra)

    def run(cid, cov, val, n, **kw):
        print(f"\n== {cid} ==")
        report(cid, run_condition(pool=pool, writer=writer, step="degradation",
                                  condition_id=cid, covariate=cov, value=val, snapshot_n=n,
                                  reps=args.reps, max_queries=args.max_queries, **kw))

    try:
        for n in args.snapshots:
            fp = fixed_p[n]
            dens = config.POISON_DENSITY[n]
            if args.quick:
                dens = [0, 2, 10]
            if "density" in args.covariates:
                for p in dens:
                    run(f"density_n{n}_p{p}", "density", p, n, p=p)
            if "similarity" in args.covariates:
                for vi, t in enumerate(V):
                    if args.quick and vi not in (0, 5):
                        continue
                    if args.only_extra and vi < 6:
                        continue
                    run(f"similarity_n{n}_v{vi}", "similarity", round(sims[vi], 4), n,
                        p=fp, trigger_text=t)
            if "shortening" in args.covariates:
                for s in (config.SHORTENING if not args.quick else [0.0, 1.0]):
                    run(f"shortening_n{n}_s{s}", "shortening", s, n, p=fp, shortening_degree=s)
            if "k" in args.covariates:
                for k in (config.K_VALUES if not args.quick else [2, 8]):
                    if k > n + fp:
                        continue
                    run(f"k_n{n}_k{k}", "k", k, n, p=fp, k=k)
            if "temperature" in args.covariates:
                for T in (config.TEMPERATURES if not args.quick else [0.1, 1.0]):
                    run(f"temperature_n{n}_T{T}", "temperature", T, n, p=fp, temperature=T)
    finally:
        writer.close()
    print("\nnext: python analyze.py --step 3   then   python step4_validate_covariates.py")


if __name__ == "__main__":
    main()
