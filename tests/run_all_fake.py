"""End-to-end plumbing test with the fake backbone. Runs every step in --quick
mode into results_fake/ and then the full analysis. ~1 minute, no GPU."""
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.argv = [sys.argv[0]]

from tests import fake_ollama
fake_ollama.install()

import analyze
import step1_clean_control, step2_trigger_control, step3_degradation, step4_validate_covariates
import step5_static_recovery, step6_interaction, step7_trajectory

# run inside a throwaway copy so the real results/ and figures/ are never touched
import tempfile
root = os.getcwd()
tmp = tempfile.mkdtemp(prefix="fakerun_")
for item in ("resilience", "data", "tests", "analyze.py", "rescore.py", "make_records.py",
             "step1_clean_control.py", "step2_trigger_control.py", "step3_degradation.py",
             "step4_validate_covariates.py", "step5_static_recovery.py", "step6_interaction.py",
             "step7_trajectory.py"):
    src = os.path.join(root, item)
    if os.path.isdir(src):
        shutil.copytree(src, os.path.join(tmp, item))
    elif os.path.exists(src):
        shutil.copy(src, tmp)
os.chdir(tmp)
print("fake run directory:", tmp)
if not os.path.exists("data/clean_records.csv"):
    import make_records  # noqa

def run(mod, *argv):
    sys.argv = [mod.__name__] + list(argv)
    print(f"\n\n######## {mod.__name__} {' '.join(argv)}")
    mod.main()

run(step1_clean_control, "--quick")
run(step2_trigger_control, "--quick")
run(step2_trigger_control, "--only_extra", "--reps", "1", "--max_queries", "5", "--snapshots", "10")
run(step3_degradation, "--reps", "2", "--max_queries", "12", "--snapshots", "10", "50")
sys.argv = ["a"]; step4_validate_covariates.main()
run(step5_static_recovery, "--quick")
run(step6_interaction, "--quick")
run(step7_trajectory, "--quick")
run(step7_trajectory, "--quick", "--tag", "v2", "--queries_per_step", "6")

import rescore
sys.argv = ["rescore"]; rescore.__name__  # imported for the check below
for p in sorted(__import__("glob").glob("results/*/trials.csv")):
    rescore.rescore_file(p, dry=False)
rescore.rescore_steps(dry=False)
analyze.BOOT = 30
for s in analyze.STEPS:
    analyze.STEPS[s]()
analyze.TAG7 = "v2"; analyze.step7()
print("\n\nFAKE END-TO-END OK  (outputs in", tmp, "— safe to delete)")
