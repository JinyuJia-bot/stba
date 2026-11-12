#!/usr/bin/env bash
# Reproduce the recorded dense STBA experiment from the repository's real inputs.
# No argument means help only. An explicitly selected stage may overwrite its
# generated artifacts; the immutable RoMa cache rejects mismatched fingerprints.
set -Eeuo pipefail

usage() {
    cat <<'EOF'
Usage: bash experiments/dense_stba/reproduce.sh STAGE [STAGE ...]

No arguments / --help: print this help and make no changes.
full: run all stages below, in dependency order (GPU and local data required).
Selecting a stage explicitly reruns it and may replace its generated results.
No packages or checkpoints are installed/downloaded by this script.

Stages (later stages require artifacts from earlier stages):
  data        Generate all five synthetic cases and both corgi [200,260) inputs.
  frontend    RoMa + temporal tracking + multi-anchor association, stride 4.
  baseline    All synthetic B0/B1/B2/B3; real B0/B1/B3 (B2 is synthetic oracle).
  b4          LSM ablation B4 on fractional synthetic and both real cases.
  density     Real stride-1 CPU frontend, then B3 in results_density1.
  stability   Independent real [0,40) and [20,60) observation windows, B3.
  ablations   Fractional: no_motion, motion_x10, uniform_observation_weights;
              integer_aligned: integer_wide_bounds. Each runs B3 separately.
  evaluate    Recompute metrics in every completed experiment directory.
  visualize   Generate B3 plots/GIFs; density cloud display capped at 3500.
  compare     Matched held-out reference-camera comparison + window diagnostics.
  audit       Replay support, fixed-camera projection, bounds, splits, hashes.
  tests       Run CPU unittest suites for dense STBA and reused RoMa/LSM helpers.

Environment overrides:
  DENSE_PY=/data/tools/miniconda3/envs/roma_corr/bin/python
  CUDA_VISIBLE_DEVICES=5
  DENSE_LOG_DIR=experiments/dense_stba/logs/reproduce/<UTC timestamp>-<PID>

Examples:
  bash experiments/dense_stba/reproduce.sh --help
  CUDA_VISIBLE_DEVICES=5 bash experiments/dense_stba/reproduce.sh full
  bash experiments/dense_stba/reproduce.sh evaluate visualize compare audit tests
  DENSE_PY=/path/to/python CUDA_VISIBLE_DEVICES=0 bash experiments/dense_stba/reproduce.sh density

Local prerequisites:
  SelfCap data: /data/datasets/selfcap/corgi-release (videos/images_2, calibration,
  sync.json), official /data/datasets/SelfCap/README.md, and pretrained checkpoints:
  /home/jjy/.cache/torch/hub/checkpoints/{roma_outdoor,dinov2_vitl14_pretrain}.pth.
  data creates the fixed-calibration roma_lsm/data_manifest.json if absent.
  The source camera intrinsics/extrinsics are never optimized. Results for real
  data are diagnostics; official sync is an external reference, not verified GT.

Canonical outputs stay under experiments/dense_stba/{data,cache,results,
cache_density1,results_density1,stability,ablations,real_comparison}.
Density uses separate observations and reuses only native RoMa/flow caches.
Independent window estimates are reported separately, never stitched together.
Full runs are not promised bitwise identical across GPU/library versions;
environment.json records the original local environment and source/weight hashes.
EOF
}

if [[ $# -eq 0 ]]; then usage; exit 0; fi
for stage in "$@"; do
    case "$stage" in
        --help|-h|help) usage; exit 0 ;;
        full|data|frontend|baseline|b4|density|stability|ablations|evaluate|visualize|compare|audit|tests) ;;
        *) printf 'Unknown stage: %s\n' "$stage" >&2; usage >&2; exit 2 ;;
    esac
done
if [[ " $* " == *' full '* && $# -ne 1 ]]; then
    printf 'Use full alone, or list individual stages.\n' >&2; exit 2
fi

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$ROOT/../.." && pwd)"
cd -- "$REPO"
DENSE_PY="${DENSE_PY:-/data/tools/miniconda3/envs/roma_corr/bin/python}"
if ! command -v "$DENSE_PY" >/dev/null 2>&1; then
    printf 'Python not found: %s; set DENSE_PY to the prepared environment.\n' "$DENSE_PY" >&2
    exit 1
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-5}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
LOG_DIR="${DENSE_LOG_DIR:-$ROOT/logs/reproduce/$(date -u +%Y%m%dT%H%M%SZ)-$$}"
mkdir -p -- "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/run.log") 2>&1
trap 'printf "Failed at reproduce.sh line %s; see %s/run.log\n" "$LINENO" "$LOG_DIR" >&2' ERR

run() { printf '\n+'; printf ' %q' "$@"; printf '\n'; "$@"; }
module() { run "$DENSE_PY" -m "experiments.dense_stba.$1" "${@:2}"; }
need_gpu() {
    run "$DENSE_PY" -c 'import os, torch; assert torch.cuda.is_available(), "CUDA unavailable; select an accessible GPU with CUDA_VISIBLE_DEVICES"; print({"CUDA_VISIBLE_DEVICES":os.environ.get("CUDA_VISIBLE_DEVICES"),"gpu":torch.cuda.get_device_name(0),"torch":torch.__version__})'
}

SYNTHETIC=(fractional wide zero noisy_occlusion near_static)
REAL=(raw integer_aligned)
RESULT_DIRS=()
for name in "${SYNTHETIC[@]}"; do RESULT_DIRS+=("$ROOT/results/synthetic_$name"); done
for name in "${REAL[@]}"; do
    RESULT_DIRS+=("$ROOT/results/corgi_$name" "$ROOT/results_density1/corgi_$name")
    RESULT_DIRS+=("$ROOT/stability/window_0_40/corgi_$name" "$ROOT/stability/window_20_60/corgi_$name")
done
for name in no_motion motion_x10 uniform_observation_weights; do
    RESULT_DIRS+=("$ROOT/ablations/$name/synthetic_fractional")
done
RESULT_DIRS+=("$ROOT/ablations/integer_wide_bounds/corgi_integer_aligned")

stage_data() {
    module synthetic_dynamic --out "$ROOT/data/synthetic" --frames 32 --fps 30 \
        --size 192 144 --seed 20261111 --cases "${SYNTHETIC[@]}"
    if [[ ! -f "$ROOT/../roma_lsm/data_manifest.json" ]]; then
        run "$DENSE_PY" -m experiments.roma_lsm.prepare_data \
            --root /data/datasets/selfcap/corgi-release --out "$ROOT/../roma_lsm"
    fi
    module data_loader --out "$ROOT/data/real" --start 200 --stop 260
}

stage_frontend() {
    need_gpu
    for name in "${SYNTHETIC[@]}"; do
        module cross_view_association "$ROOT/data/synthetic/$name/manifest.json" \
            --out "$ROOT/cache/synthetic_$name" --device cuda --stride 4
    done
    for name in "${REAL[@]}"; do
        module cross_view_association "$ROOT/data/real/$name/manifest.json" \
            --out "$ROOT/cache/corgi_$name" --device cuda --stride 4
        module data_frontend_diagnostics "$ROOT/cache/corgi_$name"
    done
}

stage_baseline() {
    need_gpu
    module run_experiment --selection all --config "$ROOT/config.json" \
        --methods B0 B1 B2 B3 --skip-frontend --device cuda \
        --cache-root "$ROOT/cache" --results-root "$ROOT/results" --evaluate
}

stage_b4() {
    need_gpu
    for selection in fractional real; do
        module run_experiment --selection "$selection" --config "$ROOT/config.json" \
            --methods B4 --skip-frontend --device cuda --cache-root "$ROOT/cache" \
            --results-root "$ROOT/results" --evaluate
    done
}

stage_density() {
    module prepare_density --source-cache "$ROOT/cache" --output-cache "$ROOT/cache_density1"
    need_gpu
    module run_experiment --selection real --config "$ROOT/configs/density1.json" \
        --methods B3 --skip-frontend --device cuda --cache-root "$ROOT/cache_density1" \
        --results-root "$ROOT/results_density1" --evaluate
}

stage_stability() {
    need_gpu
    for window in '0 40' '20 60'; do
        read -r first stop <<< "$window"
        module run_experiment --selection real --config "$ROOT/config.json" \
            --methods B3 --skip-frontend --device cuda --cache-root "$ROOT/cache" \
            --window "$first" "$stop" --results-root "$ROOT/stability/window_${first}_${stop}" --evaluate
    done
}

stage_ablations() {
    need_gpu
    for name in no_motion motion_x10 uniform_observation_weights; do
        module run_experiment --selection fractional --config "$ROOT/configs/$name.json" \
            --methods B3 --skip-frontend --device cuda --cache-root "$ROOT/cache" \
            --results-root "$ROOT/ablations/$name" --evaluate
    done
    module run_experiment --selection integer_aligned --config "$ROOT/configs/integer_wide_bounds.json" \
        --methods B3 --skip-frontend --device cuda --cache-root "$ROOT/cache" \
        --results-root "$ROOT/ablations/integer_wide_bounds" --evaluate
}

stage_evaluate() {
    local count=0 directory
    for directory in "${RESULT_DIRS[@]}"; do
        if [[ -f "$directory/meta.json" && -f "$directory/B3.npz" ]]; then
            module evaluate "$directory"
            count=$((count + 1))
        else printf 'Skip incomplete results: %s\n' "$directory"; fi
    done
    [[ $count -gt 0 ]] || { printf 'No completed results to evaluate.\n' >&2; return 1; }
}

stage_visualize() {
    local count=0 directory
    for directory in "${RESULT_DIRS[@]}"; do
        if [[ -f "$directory/meta.json" && -f "$directory/B3.npz" ]]; then
            if [[ "$directory" == "$ROOT/results_density1/"* ]]; then
                module visualize "$directory" --method B3 --max-gif-frames 12 \
                    --max-cloud-points 3500 --cloud-rgb --max-trajectory-lines 12
            else module visualize "$directory" --method B3 --max-gif-frames 12; fi
            count=$((count + 1))
        else printf 'Skip incomplete results: %s\n' "$directory"; fi
    done
    [[ $count -gt 0 ]] || { printf 'No completed results to visualize.\n' >&2; return 1; }
}

stage_compare() {
    module compare_real --results-root "$ROOT/results" --results-root "$ROOT/results_density1" \
        --stability-root "$ROOT/stability" --output "$ROOT/real_comparison"
}

stage_audit() {
    module audit_results --roots "$ROOT/results" "$ROOT/results_density1" \
        "$ROOT/stability" "$ROOT/ablations" --output "$ROOT/results_audit.json"
}

stage_tests() {
    run "$DENSE_PY" -m unittest discover -s "$ROOT/tests" -p 'test_*.py' -v
    run "$DENSE_PY" -m unittest discover -s "$ROOT/../roma_lsm/tests" -p 'test_*.py' -v
    run "$DENSE_PY" -c 'from experiments.roma_lsm.roma_adapter import adapter_self_test; print(adapter_self_test())'
}

STAGES=("$@")
if [[ ${STAGES[0]} == full ]]; then
    STAGES=(data frontend baseline b4 density stability ablations evaluate visualize compare audit tests)
fi
printf 'Repository: %s\nPython: %s\nLog directory: %s\n' "$REPO" "$DENSE_PY" "$LOG_DIR"
for stage in "${STAGES[@]}"; do
    printf '\n===== stage %s =====\n' "$stage"
    "stage_$stage"
done
printf '\nCompleted requested stages. Log: %s/run.log\n' "$LOG_DIR"
