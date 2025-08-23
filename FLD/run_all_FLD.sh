#!/usr/bin/env bash
set -euo pipefail

### FLD ###
patience=10
gpu=0

for seed in {1..5}
do
  # PhysioNet (24h history)
  python train_FLD.py \
    -dset physionet -ot 24 \
    -bs 32 -lr 1e-3 -es $patience -s $seed --gpu $gpu -fn C

  # MIMIC (24h history)
  python train_FLD.py \
    -dset mimic -ot 24 \
    -bs 32 -lr 1e-3 -es $patience -s $seed --gpu $gpu -fn C

  # Human Activity (3000 ms history)
  python train_FLD.py \
    -dset activity -ot 3000 \
    -bs 32 -lr 1e-3 -es $patience -s $seed --gpu $gpu -fn C

  # USHCN (24 months history)
  python train_FLD.py \
    -dset ushcn -ot 24 \
    -bs 192 -lr 1e-3 -es $patience -s $seed --gpu $gpu -fn C
done
