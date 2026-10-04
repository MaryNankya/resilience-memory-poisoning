#!/usr/bin/env python3
"""
Step 1 — clean control.  p = 0, NO trigger phrase.

The task-correctness ceiling, at the fixed operating point (k=4, T=0.1), at
every snapshot. Drawn as the dashed reference line on every degradation
figure. Also runs the k and T grid once at p=0 so the operating-point choice
is documented in the same results file.

Budget (5 reps, 50 queries):  3 snapshots x (1 + 4 k-values + 4 T-values)
                              x 50 x 5  ~= 5,900 trials  (n=10 uses 10 queries)
Run:   python step1_clean_control.py            (~1.5 h at 1 s/trial)
       python step1_clean_control.py --quick    (smoke test)
"""
from resilience import config
from resilience.common import std_parser, setup
from resilience.runner import run_condition, report


def main():
    ap = std_parser(__doc__)
    ap.add_argument("--no_grid", action="store_true", help="only the operating point, skip k/T grid")
    args = ap.parse_args()
    pool, writer, _ = setup(args, "step1_clean_control")
    try:
        for n in args.snapshots:
            cid = f"clean_n{n}"
            print(f"\n== {cid} (k={config.DEFAULT_K}, T={config.DEFAULT_TEMPERATURE}) ==")
            report(cid, run_condition(pool=pool, writer=writer, step="clean_control",
                                      condition_id=cid, covariate="clean", value=0,
                                      snapshot_n=n, p=0, trigger_present=False,
                                      reps=args.reps, max_queries=args.max_queries))
            if args.no_grid or args.quick:
                continue
            for k in config.K_VALUES:
                if k == config.DEFAULT_K or k > n:
                    continue
                cid = f"clean_n{n}_k{k}"
                print(f"\n== {cid} ==")
                report(cid, run_condition(pool=pool, writer=writer, step="clean_control",
                                          condition_id=cid, covariate="clean_k", value=k,
                                          snapshot_n=n, p=0, trigger_present=False, k=k,
                                          reps=args.reps, max_queries=args.max_queries))
            for T in config.TEMPERATURES:
                if T == config.DEFAULT_TEMPERATURE:
                    continue
                cid = f"clean_n{n}_T{T}"
                print(f"\n== {cid} ==")
                report(cid, run_condition(pool=pool, writer=writer, step="clean_control",
                                          condition_id=cid, covariate="clean_T", value=T,
                                          snapshot_n=n, p=0, trigger_present=False, temperature=T,
                                          reps=args.reps, max_queries=args.max_queries))
    finally:
        writer.close()
    print("\nnext: python step2_trigger_control.py")


if __name__ == "__main__":
    main()
