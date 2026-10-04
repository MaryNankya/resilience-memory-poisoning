#!/usr/bin/env bash
# =====================================================================
# Additional runs A–E, as four concurrent chains on the private Ollama
# instance (127.0.0.1:11435, GPU 3, OLLAMA_NUM_PARALLEL=4).
#
#   chain A  (llama3.1:8b, existing tree):   n=10 top-up to 25 reps in
#            steps 1,2,3,5; the six non-policy phrasings in steps 2 and 3;
#            then re-analyse with bootstrap tau.
#   chain B  (llama3.1:8b, new folder):      trajectory v2, 20 episodes,
#            16 queries/step -> results/step7_trajectory_v2
#   chain C  (qwen3:8b,      results_qwen3):  full protocol
#   chain D  (llama3.2:3b,   results_llama32): full protocol
#
# Before running:  pull the two extra backbones into the private instance
#   OLLAMA_HOST=127.0.0.1:11435 ollama pull qwen3:8b
#   OLLAMA_HOST=127.0.0.1:11435 ollama pull llama3.2:3b
# Usage:  chmod +x run_additional.sh && nohup ./run_additional.sh &
#         tail -f chainA2.log chainB2.log chainC2.log chainD2.log
# Every chain resumes if interrupted; re-run the same command.
# =====================================================================
set -u
cd "$(dirname "$0")"
[ -f .venv/bin/activate ] && source .venv/bin/activate || [ -f /x1/mnankya/.venv/bin/activate ] && # activate a virtualenv if one is present (local .venv first, then the server path used in the paper)
[ -f .venv/bin/activate ] && source .venv/bin/activate
[ -f /x1/mnankya/.venv/bin/activate ] && source /x1/mnankya/.venv/bin/activate
export OLLAMA_URL=http://127.0.0.1:11435

# ---------- chain A: llama3.1:8b top-ups in the existing results tree ----------
nohup bash -c '
  set -e
  echo "=== chain A start $(date)"
  # A. n=10 to 25 replications (runner skips the 5 already done)
  python step1_clean_control.py     --reps 25 --snapshots 10
  python step2_trigger_control.py   --reps 25 --snapshots 10
  python step3_degradation.py       --reps 25 --snapshots 10
  python step5_static_recovery.py   --reps 25 --snapshots 10 --fixed_p 1
  # E. six non-policy phrasings, all snapshots, 5 reps (25 at n=10 for consistency)
  python step2_trigger_control.py   --only_extra
  python step2_trigger_control.py   --only_extra --reps 25 --snapshots 10
  python step3_degradation.py       --only_extra --covariates similarity
  python step3_degradation.py       --only_extra --covariates similarity --reps 25 --snapshots 10
  # C. re-analyse everything with bootstrap intervals for tau (200 draws)
  python analyze.py --all --boot 200
  python step4_validate_covariates.py
  echo "=== chain A done $(date)"
' > chainA2.log 2>&1 &

# ---------- chain B: trajectory v2 ----------
nohup bash -c '
  set -e
  echo "=== chain B start $(date)"
  python step7_trajectory.py --tag v2 --episodes 20 --queries_per_step 16
  python analyze.py --step 7 --tag7 v2
  echo "=== chain B done $(date)"
' > chainB2.log 2>&1 &

# ---------- chain C: qwen3:8b full protocol ----------
BACKBONE_MODEL=qwen3:8b RESULTS_ROOT=results_qwen3 FIGURES_ROOT=figures_qwen3 STUDY_VERSION=fresh-v1-qwen3 \
nohup bash -c '
  set -e
  echo "=== chain C start $(date)  $BACKBONE_MODEL"
  python step1_clean_control.py && python analyze.py --step 1 --boot 0
  python step2_trigger_control.py && python analyze.py --step 2 --boot 0
  python step3_degradation.py && python analyze.py --step 3 --boot 0 && python step4_validate_covariates.py
  FP=$(python -c "import json;from resilience import config;r=json.load(open(f\"{config.RESULTS_ROOT}/step3_degradation/recommended_fixed_p.json\"));print(\" \".join(str(r.get(str(n)) or config.FIXED_P[n]) for n in config.SNAPSHOTS))")
  python step5_static_recovery.py --fixed_p $FP && python analyze.py --step 5 --boot 0
  python step6_interaction.py && python analyze.py --step 6 --boot 0
  python step7_trajectory.py && python analyze.py --step 7 --boot 0
  python analyze.py --all --boot 200
  echo "=== chain C done $(date)"
' > chainC2.log 2>&1 &

# ---------- chain D: llama3.2:3b full protocol ----------
BACKBONE_MODEL=llama3.2:3b RESULTS_ROOT=results_llama32 FIGURES_ROOT=figures_llama32 STUDY_VERSION=fresh-v1-llama32 \
nohup bash -c '
  set -e
  echo "=== chain D start $(date)  $BACKBONE_MODEL"
  python step1_clean_control.py && python analyze.py --step 1 --boot 0
  python step2_trigger_control.py && python analyze.py --step 2 --boot 0
  python step3_degradation.py && python analyze.py --step 3 --boot 0 && python step4_validate_covariates.py
  FP=$(python -c "import json;from resilience import config;r=json.load(open(f\"{config.RESULTS_ROOT}/step3_degradation/recommended_fixed_p.json\"));print(\" \".join(str(r.get(str(n)) or config.FIXED_P[n]) for n in config.SNAPSHOTS))")
  python step5_static_recovery.py --fixed_p $FP && python analyze.py --step 5 --boot 0
  python step6_interaction.py && python analyze.py --step 6 --boot 0
  python step7_trajectory.py && python analyze.py --step 7 --boot 0
  python analyze.py --all --boot 200
  echo "=== chain D done $(date)"
' > chainD2.log 2>&1 &

sleep 5
echo "started 4 chains; logs: chainA2.log chainB2.log chainC2.log chainD2.log"
ps aux | grep -E "step[0-9]" | grep -v grep | wc -l
