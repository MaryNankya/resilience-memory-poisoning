#!/usr/bin/env bash
# Runs the entire study in order. Safe to re-run: every step resumes where it
# stopped. Log goes to run_all.log. Start with:  nohup ./run_all.sh &
set -euo pipefail
cd "$(dirname "$0")"
exec > >(tee -a run_all.log) 2>&1
echo "=== study start $(date)  backbone=${BACKBONE_MODEL:-llama3.1:8b} results=${RESULTS_ROOT:-results} ==="

[ -f data/clean_records.csv ] || python make_records.py
python resilience/scoring.py > /dev/null

python step1_clean_control.py
python analyze.py --step 1

python step2_trigger_control.py
python analyze.py --step 2

python step3_degradation.py
python analyze.py --step 3
python step4_validate_covariates.py

# Step 5 uses the severity the density sweep recommends (smallest p with >=50% attack),
# falling back to config.FIXED_P if a snapshot never reached 50%.
FIXED_P=$(python - <<'EOF'
import json
from resilience import config
rec = json.load(open(f"{config.RESULTS_ROOT}/step3_degradation/recommended_fixed_p.json"))
print(" ".join(str(rec.get(str(n)) or config.FIXED_P[n]) for n in config.SNAPSHOTS))
EOF
)
echo "step5 fixed_p per snapshot: $FIXED_P"
python step5_static_recovery.py --fixed_p $FIXED_P
python analyze.py --step 5

python step6_interaction.py
python analyze.py --step 6

python step7_trajectory.py
python analyze.py --step 7

echo "=== study complete $(date) ==="
