#!/usr/bin/env bash
set -euo pipefail

GPU=0
LOGDIR="runs"
REPO_ROOT="$(pwd)"
DATE_TAG="$(date +"%Y%m%d_%H%M%S")"
JOBLOG_DIR="${REPO_ROOT}/joblogs/${DATE_TAG}"
mkdir -p "${JOBLOG_DIR}"

run_at () {
  local tag="$1"; shift
  local dir="$1"; shift
  local log="${JOBLOG_DIR}/${tag}.log"
  echo "--------------------------------------------------------------------------------"
  echo "[RUN @ ${dir}] $*"
  echo "--------------------------------------------------------------------------------"
  ( cd "${dir}" && "$@" ) 2>&1 | tee "${log}"
}

if [[ -f "model/run_models.py" ]]; then
  TP_DIR="model"; TP_SCRIPT="run_models.py"
elif [[ -f "tPatchGNN/run_models.py" ]]; then
  TP_DIR="tPatchGNN"; TP_SCRIPT="run_models.py"
else
  echo "ERROR: couldn't find run_models.py (checked model/ and tPatchGNN/)." >&2
  exit 1
fi

echo "==> Starting all physionet runs (GPU=${GPU}, LOGDIR=${LOGDIR})"
echo "==> Logs: ${JOBLOG_DIR}"

run_at FLD_physionet FLD        python train_FLD.py -d physionet -ot 24 -bs 32 -e 100 -es 10 --gpu "${GPU}" --tbon --logdir "${LOGDIR}"
run_at GraFITi_physionet grafiti python train_grafiti.py -d physionet -ot 24 -bs 32 --epochs 100 --early-stop 10 --gpu "${GPU}" --encoder gratif --tbon --logdir "${LOGDIR}"
run_at mTAN_physionet    mtan    python train_mtan.py -d physionet -ot 24 -bs 32 --niters 10 --early-stop 10 --gpu "${GPU}" --latent-dim 16 --rec-hidden 32 --gen-hidden 50 --embed-time 128 --k-iwae 10 --lr 1e-3 --tbon --logdir "${LOGDIR}"
run_at tPatchGNN_physionet "${TP_DIR}" python "${TP_SCRIPT}" --dataset physionet --history 24 --patch_size 8 --stride 8 --batch_size 32 --epoch 100 --patience 10 --gpu "${GPU}" --tbon --logdir "${LOGDIR}"

echo "==> All jobs finished. Logs in ${JOBLOG_DIR}"
echo "Open TensorBoard:  tensorboard --logdir ${LOGDIR}"
