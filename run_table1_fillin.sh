#!/bin/bash
# Fill in missing data points for paper Table 1 (LLaMA1-7B column).
#
# Tasks:
#   1) PALS @ gs=0.6 with probe_sparsity=0.6 (probe_seed=1, eval_seed=1)
#   2) Uniform-WANDA @ gs in {0.5, 0.6, 0.7} for seed in {0, 1}
#
# OOM-aware: each task auto-retries up to 3 times with a 10-minute cool-down
# whenever the failure trace contains "out of memory" / "CUDA error".
#
# Single GPU only (default GPU 0). Override with GPU=1.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GPU="${GPU:-0}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-owl}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/2T/zhuhe/data/decapoda-research-llama-7B-hf}"
TOKENIZER_NAME_OR_PATH="${TOKENIZER_NAME_OR_PATH:-$MODEL_NAME_OR_PATH}"
CACHE_DIR="${CACHE_DIR:-/2T/zhuhe/results/llm_weights}"
DATASET="${DATASET:-c4}"

PALS_RESULTS_ROOT="${PALS_RESULTS_ROOT:-/2T/zhuhe/results/probe_sparsity_downstream_area_ppl_sweep}"
UNIFORM_ROOT="${UNIFORM_ROOT:-/2T/zhuhe/results/table1_fillin/uniform_wanda}"
LOG_DIR="${LOG_DIR:-/2T/zhuhe/results/table1_fillin/logs}"

mkdir -p "$LOG_DIR" "$UNIFORM_ROOT"

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

is_oom_log() {
    local logf="$1"
    grep -qiE "out of memory|CUDA error|CUDA out of memory|cudaErrorMemoryAllocation" "$logf" 2>/dev/null
}

run_with_oom_retry() {
    local label="$1"; shift
    local logf="$1"; shift
    local max_retries="${MAX_RETRIES:-3}"
    local retry=0
    while (( retry <= max_retries )); do
        echo "[$(date '+%F %T')] === $label === (attempt $((retry+1))/$((max_retries+1)))"
        ( "$@" ) 2>&1 | tee "$logf"
        local rc=${PIPESTATUS[0]}
        if [[ $rc -eq 0 ]]; then
            echo "[$(date '+%F %T')] [$label] succeeded"
            return 0
        fi
        if is_oom_log "$logf"; then
            retry=$((retry+1))
            echo "[$(date '+%F %T')] [$label] OOM detected, sleeping 600s then retrying (retry=$retry)"
            sleep 600
            continue
        fi
        echo "[$(date '+%F %T')] [$label] non-OOM failure (rc=$rc); aborting this task"
        return $rc
    done
    echo "[$(date '+%F %T')] [$label] giving up after $max_retries retries"
    return 1
}

# ----------------------------------------------------------------------------
# Task 1: PALS @ gs=0.6, probe=0.6, probe_seed=1, eval_seed=1
# ----------------------------------------------------------------------------

task_pals_gs06() {
    local label="pals_gs06_probe06_seed1"
    local logf="$LOG_DIR/${label}.log"
    local result_csv="$PALS_RESULTS_ROOT/probe_0p6/submission/gs_0.6/seed_1/downstream_area/one_line_result.csv"
    if [[ -f "$result_csv" ]]; then
        echo "[skip] $label already done -> $result_csv"
        return 0
    fi

    run_with_oom_retry "$label" "$logf" \
        env \
            PROBE_SPARSITIES_CSV=0.6 \
            GLOBAL_SPARSITY=0.6 \
            SEED=1 \
            PROBE_SEED=1 \
            GPU_DEVICES_CSV="$GPU" \
            RESULTS_ROOT="$PALS_RESULTS_ROOT" \
            SKIP_EXISTING=true \
            CONDA_ENV_NAME="$CONDA_ENV_NAME" \
        bash "$SCRIPT_DIR/run_probe_sparsity_sweep.sh"
}

# ----------------------------------------------------------------------------
# Task 2: Uniform-WANDA via run_dependency_submission_gate.py
#         (score_type=uniform, alpha=0 -> exact uniform schedule)
# ----------------------------------------------------------------------------

ALPHA_JSON="$UNIFORM_ROOT/alpha_map.json"
echo '{"uniform": 0.0}' > "$ALPHA_JSON"

run_uniform_one() {
    local gs="$1"
    local seed="$2"
    local gs_tag
    gs_tag="$(echo "$gs" | tr . p)"
    local save_dir="$UNIFORM_ROOT/gs_${gs_tag}/seed_${seed}"
    local label="uniform_gs${gs_tag}_seed${seed}"
    local logf="$LOG_DIR/${label}.log"
    if [[ -f "$save_dir/one_line_result.csv" ]]; then
        echo "[skip] $label already done"
        return 0
    fi
    mkdir -p "$save_dir"

    run_with_oom_retry "$label" "$logf" \
        env CUDA_VISIBLE_DEVICES="$GPU" \
        conda run -n "$CONDA_ENV_NAME" python \
            "$SCRIPT_DIR/run_dependency_submission_gate.py" \
            --model_name_or_path "$MODEL_NAME_OR_PATH" \
            --tokenizer_name_or_path "$TOKENIZER_NAME_OR_PATH" \
            --dataset "$DATASET" \
            --global_sparsity "$gs" \
            --score_type uniform \
            --alpha_map_json "$ALPHA_JSON" \
            --seed "$seed" \
            --nsamples 128 \
            --seqlen 2048 \
            --device cuda:0 \
            --save_dir "$save_dir" \
            --eval_zeroshot false \
            --shuffle_seed "$seed" \
            --generator_mode rank_logistic \
            --profile_temperature 0.15 \
            --dry_run_schedule_only false \
            --debug_schedule true \
            --stream_batch_size 4 \
            --cache_dir "$CACHE_DIR"
}

task_uniform_all() {
    for gs in 0.5 0.6 0.7; do
        for seed in 0 1; do
            run_uniform_one "$gs" "$seed" || echo "[warn] uniform gs=$gs seed=$seed failed; continuing"
        done
    done
}

# ----------------------------------------------------------------------------
# Aggregate at the end
# ----------------------------------------------------------------------------

aggregate() {
    local out="$LOG_DIR/table1_summary.csv"
    {
        echo "method,global_sparsity,seed,ppl_wikitext2,actual_sparsity,save_dir"

        # PALS gs=0.5 probe=0.5 seed=1
        local f
        f="$PALS_RESULTS_ROOT/probe_0p5/submission/gs_0.5/seed_1/downstream_area/eval_results.json"
        [[ -f "$f" ]] && python3 -c "import json;e=json.load(open('$f'));print(f\"PALS_probe_eq_gs,0.5,1,{e['ppl_wikitext2']:.4f},{e['actual_sparsity']:.4f},$f\")"

        # PALS gs=0.6 probe=0.6 seed=1
        f="$PALS_RESULTS_ROOT/probe_0p6/submission/gs_0.6/seed_1/downstream_area/eval_results.json"
        [[ -f "$f" ]] && python3 -c "import json;e=json.load(open('$f'));print(f\"PALS_probe_eq_gs,0.6,1,{e['ppl_wikitext2']:.4f},{e['actual_sparsity']:.4f},$f\")"

        # PALS gs=0.7 probe=0.7 seed=1
        f="$PALS_RESULTS_ROOT/probe_0p7/submission/gs_0.7/seed_1/downstream_area/eval_results.json"
        [[ -f "$f" ]] && python3 -c "import json;e=json.load(open('$f'));print(f\"PALS_probe_eq_gs,0.7,1,{e['ppl_wikitext2']:.4f},{e['actual_sparsity']:.4f},$f\")"

        for gs_tag in 0p5 0p6 0p7; do
            for seed in 0 1; do
                f="$UNIFORM_ROOT/gs_${gs_tag}/seed_${seed}/eval_results.json"
                [[ -f "$f" ]] && python3 -c "import json;e=json.load(open('$f'));gs='$gs_tag'.replace('p','.');print(f\"Uniform_WANDA,{gs},$seed,{e['ppl_wikitext2']:.4f},{e['actual_sparsity']:.4f},$f\")"
            done
        done
    } > "$out"
    echo "=== Aggregated summary ==="
    cat "$out"
    echo "(saved to $out)"
}

# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------

echo "[$(date '+%F %T')] starting table1 fill-in driver, GPU=$GPU"

task_pals_gs06
task_uniform_all
aggregate

echo "[$(date '+%F %T')] driver finished"
