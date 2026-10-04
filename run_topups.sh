#!/usr/bin/env bash
# =====================================================================
# Overnight top-ups (v3 -> v4 data), replication backbones only:
#   chain Q1  qwen3:8b   n=10 -> 25 reps (steps 1,2,3,5) + six extra phrasings
#   chain Q2  qwen3:8b   trajectory v2 (20 episodes x 16 queries/step)
#   chain G1  gemma3:12b same top-ups as Q1
#   chain G2  gemma3:12b trajectory v2
#
# Before running (once):
#   pkill -u $USER -f "ollama serve"          # stop the old private instance
#   OLLAMA_HOST=127.0.0.1:11435 CUDA_VISIBLE_DEVICES=3 OLLAMA_NUM_PARALLEL=4 \
#   OLLAMA_CONTEXT_LENGTH=8192 OLLAMA_MAX_LOADED_MODELS=4 \
#   OLLAMA_MODELS=/x1/mnankya/ollama_models nohup ollama serve > ollama_private.log 2>&1 &
#
# Usage:  chmod +x run_topups.sh && nohup ./run_topups.sh > topups.log 2>&1 &
# Progress:  tail -n 3 chainQ1.log chainQ2.log chainG1.log chainG2.log
# Every chain resumes if interrupted; re-run the same command.
# =====================================================================
set -u
cd "$(dirname "$0")"
[ -f .venv/bin/activate ] && source .venv/bin/activate || [ -f /x1/mnankya/.venv/bin/activate ] && # activate a virtualenv if one is present (local .venv first, then the server path used in the paper)
[ -f .venv/bin/activate ] && source .venv/bin/activate
[ -f /x1/mnankya/.venv/bin/activate ] && source /x1/mnankya/.venv/bin/activate
export OLLAMA_URL=http://127.0.0.1:11435

topups() {   # $1 = model  $2 = results root  $3 = figures root  $4 = version  $5 = fixed_p at n=10
  BACKBONE_MODEL=$1 RESULTS_ROOT=$2 FIGURES_ROOT=$3 STUDY_VERSION=$4 bash -c '
    set -e
    echo "=== top-ups start $(date) $BACKBONE_MODEL"
    python step1_clean_control.py   --reps 25 --snapshots 10
    python step2_trigger_control.py --reps 25 --snapshots 10
    python step2_trigger_control.py --only_extra
    python step2_trigger_control.py --only_extra --reps 25 --snapshots 10
    python step3_degradation.py     --reps 25 --snapshots 10
    python step3_degradation.py     --only_extra --covariates similarity
    python step3_degradation.py     --only_extra --covariates similarity --reps 25 --snapshots 10
    python step5_static_recovery.py --reps 25 --snapshots 10 --fixed_p '"$5"'
    python analyze.py --all --boot 200
    python step4_validate_covariates.py
    echo "=== top-ups done $(date) $BACKBONE_MODEL"
  '
}
trajv2() {   # $1 = model  $2 = results root  $3 = figures root  $4 = version
  BACKBONE_MODEL=$1 RESULTS_ROOT=$2 FIGURES_ROOT=$3 STUDY_VERSION=$4 bash -c '
    set -e
    echo "=== trajectory v2 start $(date) $BACKBONE_MODEL"
    python step7_trajectory.py --tag v2 --episodes 20 --queries_per_step 16
    python analyze.py --step 7 --tag7 v2 --boot 0
    echo "=== trajectory v2 done $(date) $BACKBONE_MODEL"
  '
}

nohup bash -c "$(declare -f topups); topups qwen3:8b  results_qwen3  figures_qwen3  fresh-v1-qwen3  1" > chainQ1.log 2>&1 &
nohup bash -c "$(declare -f trajv2); trajv2 qwen3:8b  results_qwen3  figures_qwen3  fresh-v1-qwen3"    > chainQ2.log 2>&1 &
nohup bash -c "$(declare -f topups); topups gemma3:12b results_gemma3 figures_gemma3 fresh-v1-gemma3 1" > chainG1.log 2>&1 &
nohup bash -c "$(declare -f trajv2); trajv2 gemma3:12b results_gemma3 figures_gemma3 fresh-v1-gemma3"   > chainG2.log 2>&1 &

sleep 5
echo "started 4 chains; logs: chainQ1.log chainQ2.log chainG1.log chainG2.log"
ps aux | grep -E "step[0-9]" | grep -v grep | wc -l
