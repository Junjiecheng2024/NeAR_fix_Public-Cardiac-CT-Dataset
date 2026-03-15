#!/usr/bin/env bash

# Shared helpers for portable SLURM / HPC entrypoints.

near_repo_root() {
    if [[ -n "${NEAR_REPO_ROOT:-}" ]]; then
        printf '%s\n' "$NEAR_REPO_ROOT"
    else
        printf '%s\n' "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
    fi
}

near_setup_env() {
    export NEAR_REPO_ROOT="${NEAR_REPO_ROOT:-$(near_repo_root)}"
    export NEAR_DATA_ROOT="${NEAR_DATA_ROOT:-${NEAR_REPO_ROOT}/dataset}"
    export NEAR_OUTPUT_ROOT="${NEAR_OUTPUT_ROOT:-${NEAR_REPO_ROOT}/outputs}"
    export NEAR_LOG_ROOT="${NEAR_LOG_ROOT:-${NEAR_OUTPUT_ROOT}/logs}"
    export NEAR_CACHE_ROOT="${NEAR_CACHE_ROOT:-${NEAR_OUTPUT_ROOT}/cache}"
    export NEAR_PHASE1_CHECKPOINT_ROOT="${NEAR_PHASE1_CHECKPOINT_ROOT:-${NEAR_OUTPUT_ROOT}/phase1/checkpoints}"
    export NEAR_PHASE3_EVAL_CSV="${NEAR_PHASE3_EVAL_CSV:-${NEAR_OUTPUT_ROOT}/phase3/evaluation_results_full.csv}"
    export NEAR_PYTHON_BIN="${NEAR_PYTHON_BIN:-python3}"
    export NEAR_APPTAINER_BIN="${NEAR_APPTAINER_BIN:-apptainer}"
    export NEAR_HOME="${NEAR_HOME:-${HOME:-${NEAR_CACHE_ROOT}/home}}"
    export HOME="$NEAR_HOME"

    export PYTHONUSERBASE="${PYTHONUSERBASE:-${NEAR_CACHE_ROOT}/pyuser}"
    export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${NEAR_CACHE_ROOT}/pip-cache}"
    export TMPDIR="${TMPDIR:-${NEAR_CACHE_ROOT}/tmp}"
    export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${NEAR_CACHE_ROOT}/xdg-cache}"
    export WANDB_DIR="${WANDB_DIR:-${NEAR_CACHE_ROOT}/wandb}"
    export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-${NEAR_CACHE_ROOT}/wandb-config}"
    export MPLCONFIGDIR="${MPLCONFIGDIR:-${NEAR_CACHE_ROOT}/matplotlib}"
    export PATH="${PYTHONUSERBASE}/bin:${PATH}"

    mkdir -p \
        "${NEAR_OUTPUT_ROOT}" \
        "${NEAR_LOG_ROOT}" \
        "${NEAR_CACHE_ROOT}" \
        "${NEAR_PHASE1_CHECKPOINT_ROOT}" \
        "$(dirname "${NEAR_PHASE3_EVAL_CSV}")" \
        "${PYTHONUSERBASE}" \
        "${PIP_CACHE_DIR}" \
        "${TMPDIR}" \
        "${XDG_CACHE_HOME}" \
        "${WANDB_DIR}" \
        "${WANDB_CONFIG_DIR}" \
        "${MPLCONFIGDIR}" \
        "${HOME}"
}

near_run() {
    if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v srun >/dev/null 2>&1; then
        srun "$@"
    else
        "$@"
    fi
}

near_run_python_impl() {
    local use_nv="$1"
    shift

    if [[ -n "${NEAR_CONTAINER:-}" ]]; then
        local cmd=("${NEAR_APPTAINER_BIN}" "exec")
        local bind_paths=(
            "${NEAR_REPO_ROOT}"
            "${NEAR_DATA_ROOT}"
            "${NEAR_OUTPUT_ROOT}"
            "${PYTHONUSERBASE}"
            "${PIP_CACHE_DIR}"
            "${TMPDIR}"
            "${XDG_CACHE_HOME}"
            "${WANDB_DIR}"
            "${WANDB_CONFIG_DIR}"
            "${MPLCONFIGDIR}"
            "${HOME}"
            "${NEAR_GT_ROOT:-}"
        )

        if [[ "${use_nv}" == "1" ]]; then
            cmd+=(--nv)
        fi

        for bind_path in "${bind_paths[@]}"; do
            if [[ -z "${bind_path}" ]]; then
                continue
            fi
            mkdir -p "${bind_path}"
            cmd+=(-B "${bind_path}:${bind_path}")
        done

        cmd+=("${NEAR_CONTAINER}" "${NEAR_PYTHON_BIN}" "$@")
        near_run "${cmd[@]}"
    else
        near_run "${NEAR_PYTHON_BIN}" "$@"
    fi
}

near_run_python() {
    near_run_python_impl 0 "$@"
}

near_run_python_gpu() {
    near_run_python_impl 1 "$@"
}
