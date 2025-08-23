# Inter-Channel Attention for Functional Latent Dynamics in Irregularly Sampled Time Series

This repository is an extended implementation of [Functional-Latent_Dynamics](https://github.com/kloetergensc/Functional-Latent_Dynamics), developed as part of Yazdan Asadi’s Master's Thesis at the University of Hildesheim.

It introduces novel model variants based on **Inter-Channel Functional Latent Dynamics (ICFLD)**, integrating multiple attention mechanisms and residual cycle forecasting for benchmarking irregularly sampled multivariate time series forecasting on real-world datasets such as MIMIC-III, MIMIC-IV, PhysioNet 2012, and USHCN.

---

## Thesis Goal

To improve Functional Latent Dynamics (FLD) for irregular time series forecasting by introducing **Inter-Channel Attention Mechanisms**, including **Residual Cycle Forecasting (RCF)**, and benchmarking against competitive baselines.

---

## Models

| Model       | Description                                                                 |
|-------------|-----------------------------------------------------------------------------|
| `ICFLD`     | Original Functional Latent Dynamics with inter-channel attention            |
| `ICFLD-RCF` | ICFLD extended with **Residual Cycle Forecasting** (CycleNet variant)       |
| `FLD`       | Functional Latent Dynamics                                                  |
| `MTAN`      | Multi-task Attention Networks                                               |
| `GraFITi`    | Graph-structured attention model for irregular multivariate time series     |
| `t-PatchGNN`| A Transformable Patching Graph Neural Networks Approach                     |

---

### ICFLD-RCF: Residual Cycle Forecasting

`ICFLD-RCF` introduces **Residual Cycle Forecasting (RCF)** on top of ICFLD to explicitly remove and reintroduce periodic components in time series data.

- **RCF Mechanism:**  [CycleNet](https://arxiv.org/abs/2409.18479)
  Cyclical signals (e.g., circadian rhythms or seasonal cycles)  are estimated and **removed from the input** before passing through the attention layers. The model then focuses on the residual signal dynamics. At the end of the pipeline, the cycles are **added back** to reconstruct the final forecast.

- **Usage:**  
  To enable this behavior, set the `--cycle` flag when running the ICFLD model.


# Inter-Channel Attention for Functional Latent Dynamics in Irregularly Sampled Time Series

This repository extends [Functional-Latent_Dynamics](https://github.com/kloetergensc/Functional-Latent_Dynamics) with **Inter-Channel Functional Latent Dynamics (IC-FLD)** and a unified training stack for baselines including **FLD**, **mTAN**, **GraFITi**, and **t-PatchGNN**. It is part of Yazdan Asadi’s Master’s Thesis at the University of Hildesheim.

**Highlights**
- Unified preprocessing (from the [t-PatchGNN](https://github.com/usail-hkust/t-PatchGNN/tree/main) codebase) shared across all models.
- IC-FLD with **Inter-Channel Attention** and optional **Residual Cycle Forecasting (RCF)**.
- Consistent trainers, logging, and evaluation across models.
- **TensorBoard** support via `--tbon` and `--logdir`.

---

## Models

| Model          | Description                                                                 |
|----------------|-----------------------------------------------------------------------------|
| **IC-FLD**     | Inter-Channel FLD with attention over all channels jointly.                 |
| **IC-FLD-RCF** | IC-FLD with **Residual Cycle Forecasting** (remove cycles → model residuals → add cycles back). |
| **FLD**        | Original Functional Latent Dynamics baseline.                               |
| **mTAN**       | Multi-task Attention Networks baseline.                                     |
| **GraFITi**    | Graph-structured attention baseline (GrATiF encoder).                       |
| **t-PatchGNN** | Patching Graph Neural Network (original repo’s model & preprocessing).      |

### IC-FLD basis & RCF (what it does)
- **Basis functions**: choose with `-fn {C,L,Q,S}` for constant/linear/quadratic/sinusoidal bases.  
- **Inter-Channel Attention**: attention is computed over a flattened **time × channel** sequence so heads can learn cross-channel structure.  
- **Residual Cycle Forecasting (RCF)**: enable with `--use-cycle --cycle-length <period>`. The model subtracts a per-phase cycle estimate before attention, and adds it back when predicting.

---

## Notes:

- **Single preprocessing path** for all models (via `lib/parse_datasets.py`, `lib/physionet.py`, etc.).  
  - Time/value normalization and train/val/test splits are shared.
  - Non-patch models call parsers with `patch_ts=False`.
- **Consistent CLI** across trainers; all support:
  - `--tbon` to enable TensorBoard scalars; `--logdir runs` to pick a log root.
  - Dataset selection: `-d physionet|mimic|ushcn|activity`.
  - Observation window: `-ot <hours>`.
- **IC-FLD resume**: `--resume auto` continues from the most recent `*.latest.pt`.
- **Queue scripts**: run all models for PhysioNet in sequence (`run_all_physionet.sh` / `.ps1` / `.bat`).  

---
## Datasets

- **MIMIC**: De-identified ICU patient records  
- **USHCN**: U.S. Historical Climatology Network (climate data)  
- **PhysioNet**: ICU challenge dataset with multivariate physiological signals  

---
## Train one model

Run these from the model’s subfolder (e.g., `cd FLD/`), or use the queue scripts below.

### FLD
```bash
python train_FLD.py \
  -d physionet -ot 24 -bs 32 -e 100 -es 10 --gpu 0 --tbon --logdir runs
```

### IC-FLD (no cycles)
```bash
python train_FLD_ICC.py \
  -d physionet -ot 24 -bs 32 --epochs 100 --early-stop 10 \
  --gpu 0 -fn L -ed 64 -ld 64 -nh 2 \
  --tbon --logdir runs --resume auto
```

### IC-FLD-RCF (with cycles)
```bash
python train_FLD_ICC.py \
  -d physionet -ot 24 -bs 32 --epochs 100 --early-stop 10 \
  --gpu 0 -fn L -ed 64 -ld 64 -nh 2 \
  --use-cycle --cycle-length 24 --time-max-hours 48 \
  --tbon --logdir runs --resume auto
```

# Tensorboard
```bash
tensorboard --logdir runs
```