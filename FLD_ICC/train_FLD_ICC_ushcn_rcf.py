#!/usr/bin/env python
# coding: utf-8
"""
IC-FLD (USHCN) trainer with Residual Cycle Forecasting (RCF) ALWAYS ON.
- Same feel as your no-cycle USHCN trainer.
- Extra flags: --cycle-length, --time-max-hours
- Passes denorm_time_max to the model forward every time.
"""

import argparse, sys, os, time, random, warnings, inspect, glob
from random import SystemRandom
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

# ---- project root / lib discovery ----
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
try:
    import lib.utils as utils
    from lib.parse_datasets import parse_datasets
except ModuleNotFoundError:
    sys.path.append(str(REPO_ROOT))
    import lib.utils as utils
    from lib.parse_datasets import parse_datasets

# IC-FLD core
from FLD_ICC import IC_FLD

# optional shared CSV writer
try:
    from write_result import write_result
except Exception:
    write_result = None

# ---------------- CLI ----------------
p = argparse.ArgumentParser(description="IC-FLD (USHCN) with residual cycle removal (RCF)")
p.add_argument("-r", "--run_id", default=None, type=str)
p.add_argument("-e", "--epochs", default=300, type=int)
p.add_argument("-es", "--early-stop", default=30, type=int)
p.add_argument("-bs", "--batch-size", default=64, type=int)
p.add_argument("-lr", "--learn-rate", default=1e-3, type=float)
p.add_argument("-wd", "--weight-decay", default=0.0, type=float)
p.add_argument("-s", "--seed", default=0, type=int)

p.add_argument("-d", "--dataset", default="ushcn", type=str,
               help="physionet | mimic | ushcn | activity (use ushcn here)")
p.add_argument("-ot", "--observation-time", default=24, type=int, help="history window length")

# model hyperparams (parity with your no-cycle script)
p.add_argument("-fn", "--function", default="L", choices=("L", "S", "C", "Q"))
p.add_argument("-ed", "--embedding-dim", default=64, type=int)    # total embedding dim
p.add_argument("-nh", "--num-heads", default=2, type=int)
p.add_argument("-ld", "--latent-dim", default=64, type=int)
p.add_argument("-dp", "--depth", default=2, type=int)

# RCF params (ALWAYS used)
p.add_argument("--cycle-length", type=int, default=24, help="cycle length in hours")
p.add_argument("--time-max-hours", type=int, default=48, help="max hours range that timestamps were scaled to")

p.add_argument("--gpu", default="0", type=str)
p.add_argument("--resume", default="", type=str, help="'auto' or path to a .pt checkpoint")

# TensorBoard
p.add_argument("--tbon", action="store_true", help="Enable TensorBoard logging")
p.add_argument("--logdir", type=str, default="runs", help="TensorBoard log root")

args = p.parse_args()

# ---------------- Setup ----------------
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)

experiment_id = int(SystemRandom().random() * 10000000)
SAVE_DIR = THIS_DIR / "saved_models"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
ckpt_best  = SAVE_DIR / f"ICFLD-RCF-ushcn-{experiment_id}.best.pt"
ckpt_last  = SAVE_DIR / f"ICFLD-RCF-ushcn-{experiment_id}.latest.pt"

# ---------------- Data via tPatchGNN (no patches) ----------------
dataset_map = {"physionet":"physionet","p12":"physionet",
               "mimic":"mimic","mimiciii":"mimic","ushcn":"ushcn","activity":"activity"}
dataset_name = dataset_map.get(args.dataset.lower(), args.dataset)

pd_args = SimpleNamespace(
    state="def",
    n=int(1e8),
    hop=1, nhead=1, tf_layer=1, nlayer=1,
    epoch=args.epochs, patience=args.early_stop,
    history=int(args.observation_time),
    patch_size=8.0, stride=8.0,
    logmode="a",
    lr=float(args.learn_rate), w_decay=float(args.weight_decay),
    batch_size=int(args.batch_size),
    save="experiments/", load=None,
    seed=int(args.seed), dataset=dataset_name,
    quantization=0.0,
    model="IC-FLD-RCF", outlayer="Linear",
    hid_dim=64, te_dim=10, node_dim=10,
    gpu=args.gpu,
)
pd_args.npatch = int(np.ceil((pd_args.history - pd_args.patch_size) / pd_args.stride)) + 1
pd_args.device = DEVICE

data_obj = parse_datasets(pd_args, patch_ts=False)
INPUT_DIM = data_obj["input_dim"]
num_train_batches = data_obj["n_train_batches"]

print("PID, device:", os.getpid(), DEVICE)
print(f"Dataset={dataset_name}, INPUT_DIM={INPUT_DIM}, history={pd_args.history}")
print("n_train_batches:", num_train_batches)

# ---------------- Model ----------------
sig = inspect.signature(IC_FLD.__init__)
accepts = set(sig.parameters.keys())
kw = dict(
    input_dim=INPUT_DIM,
    latent_dim=args.latent_dim,
    num_heads=args.num_heads,
    embed_dim=args.embedding_dim,
    function=args.function,
)
if "depth"          in accepts: kw["depth"] = args.depth
if "use_cycle"      in accepts: kw["use_cycle"] = True                 # <— force cycle mode
if "cycle_length"   in accepts: kw["cycle_length"] = args.cycle_length
if "time_max_hours" in accepts: kw["time_max_hours"] = args.time_max_hours

MODEL = IC_FLD(**kw).to(DEVICE)

# ---------------- Helpers ----------------
def _orient_time_last(x: torch.Tensor, input_dim: int) -> torch.Tensor:
    if x.dim() != 3: raise ValueError(f"Expected 3D tensor, got {x.shape}")
    if x.shape[-1] == input_dim: return x
    if x.shape[1] == input_dim:  return x.transpose(1, 2).contiguous()
    raise ValueError(f"Cannot infer feature axis from {x.shape} with D={input_dim}")

def mse_masked(y: torch.Tensor, yhat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    y    = torch.nan_to_num(y,    nan=0.0)
    yhat = torch.nan_to_num(yhat, nan=0.0)
    m = mask.float()
    num = (m * (y - yhat) ** 2).sum()
    den = m.sum().clamp_min(1.0)
    return num / den

def _ushcn_safe_context(T: torch.Tensor, X: torch.Tensor, M: torch.Tensor):
    B, L, D = X.shape
    no_ctx = (M.view(B, -1).sum(dim=1) == 0)
    if no_ctx.any():
        idx = torch.nonzero(no_ctx).squeeze(-1)
        X[idx, 0, 0] = 0.0
        M[idx, 0, 0] = 1.0
        if L > 0: T[idx, 0] = 0.0
    return T, X, M

def batch_to_icfld(batch: dict, input_dim: int, device: torch.device):
    T  = batch["observed_tp"].to(device)                                   # [B,L]
    X  = _orient_time_last(batch["observed_data"].to(device), input_dim)   # [B,L,D]
    M  = _orient_time_last(batch["observed_mask"].to(device), input_dim)   # [B,L,D]
    TY = batch["tp_to_predict"].to(device)                                 # [B,Ty]
    Y  = _orient_time_last(batch["data_to_predict"].to(device), input_dim) # [B,Ty,D]
    YM = _orient_time_last(batch["mask_predicted_data"].to(device), input_dim) # [B,Ty,D]

    X  = torch.nan_to_num(X, nan=0.0)
    M  = torch.where(torch.isnan(X), torch.zeros_like(M), M)
    T, X, M = _ushcn_safe_context(T, X, M)
    return T, X, M, TY, Y, YM

@torch.no_grad()
def evaluate(loader, nbatches):
    total = 0.0; cnt = 0.0
    for _ in range(nbatches):
        b = utils.get_next_batch(loader)
        T, X, M, TY, Y, YM = batch_to_icfld(b, INPUT_DIM, DEVICE)
        YH = MODEL(T, X, M, TY, denorm_time_max=args.time_max_hours)
        total += float(((Y - YH) ** 2 * YM).sum().item())
        cnt   += float(YM.sum().item())
    mse = total / max(1.0, cnt)
    rmse = (mse + 1e-8) ** 0.5
    return {"loss": mse, "mse": mse, "rmse": rmse}

# ---------------- Optim / sched / TB ----------------
optimizer = optim.AdamW(MODEL.parameters(), lr=float(args.learn_rate), weight_decay=float(args.weight_decay))
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=10, factor=0.5, verbose=False)

writer = None
if args.tbon:
    from torch.utils.tensorboard import SummaryWriter
    run_name = f"ICFLD_RCF_USHCN_{int(time.time())}"
    writer = SummaryWriter(log_dir=os.path.join(args.logdir, run_name))
    try:
        b0 = utils.get_next_batch(data_obj["train_dataloader"])
        T0, X0, M0, TY0, _, _ = batch_to_icfld(b0, INPUT_DIM, DEVICE)
        writer.add_graph(MODEL, (T0, X0, M0, TY0,))  # graph without denorm arg for TB
    except Exception as e:
        print(f"[tbgraph] skipped: {e}")

# ---------------- Resume (optional) ----------------
start_epoch = 1
best_val = float("inf"); best_iter = 0; test_report = None
if args.resume:
    if args.resume == "auto":
        pattern = str(SAVE_DIR / ("ICFLD-RCF-ushcn-*.latest.pt"))
        ckpts = sorted(glob.glob(pattern), key=os.path.getmtime)
        load_path = ckpts[-1] if ckpts else ""
    else:
        load_path = args.resume
    if load_path and Path(load_path).exists():
        ckpt = torch.load(load_path, map_location="cpu")
        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        MODEL.load_state_dict(state, strict=False)
        if isinstance(ckpt, dict):
            if "optimizer" in ckpt:
                try: optimizer.load_state_dict(ckpt["optimizer"])
                except: pass
            if "epoch" in ckpt:     start_epoch = int(ckpt["epoch"]) + 1
            if "best_val" in ckpt:  best_val    = float(ckpt["best_val"])
            if "best_iter" in ckpt: best_iter   = int(ckpt["best_iter"])
        print(f"[resume] loaded {load_path} (start_epoch={start_epoch}, best_val={best_val:.6f})")

# ---------------- Train ----------------
print("Starting training (RCF)…")
for epoch in range(start_epoch, args.epochs + 1):
    t0 = time.time()
    MODEL.train()
    last_train_loss = None

    for _ in range(num_train_batches):
        optimizer.zero_grad(set_to_none=True)
        b = utils.get_next_batch(data_obj["train_dataloader"])
        T, X, M, TY, Y, YM = batch_to_icfld(b, INPUT_DIM, DEVICE)
        YH = MODEL(T, X, M, TY, denorm_time_max=args.time_max_hours)
        loss = mse_masked(Y, YH, YM)
        loss.backward(); optimizer.step()
        last_train_loss = float(loss.item())

    MODEL.eval()
    val_res = evaluate(data_obj["val_dataloader"], data_obj["n_val_batches"])

    if val_res["mse"] < best_val:
        best_val = val_res["mse"]; best_iter = epoch
        test_report = evaluate(data_obj["test_dataloader"], data_obj["n_test_batches"])
        torch.save({
            "state_dict": MODEL.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_val": best_val,
            "best_iter": best_iter,
            "args": vars(args),
            "input_dim": INPUT_DIM,
        }, ckpt_best)

    torch.save({
        "state_dict": MODEL.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_val": best_val,
        "best_iter": best_iter,
        "args": vars(args),
        "input_dim": INPUT_DIM,
    }, ckpt_last)

    scheduler.step(val_res["loss"])

    dt = time.time() - t0
    print(f"- Epoch {epoch:03d} | train_loss(one-batch): {last_train_loss:.6f} | "
          f"val_mse: {val_res['mse']:.6f} | val_rmse: {val_res['rmse']:.6f} | "
          + (f"best@{best_iter} test_mse: {test_report['mse']:.6f} rmse: {test_report['rmse']:.6f} | " if test_report else "")
          + f"time: {dt:.2f}s")

    if writer:
        writer.add_scalar("train/loss_one_batch", last_train_loss, epoch)
        writer.add_scalar("val/mse",  float(val_res["mse"]),  epoch)
        writer.add_scalar("val/rmse", float(val_res["rmse"]), epoch)
        if test_report:
            writer.add_scalar("test/mse_best",  float(test_report["mse"]),  epoch)
            writer.add_scalar("test/rmse_best", float(test_report["rmse"]), epoch)

    if (epoch - best_iter) >= args.early_stop:
        print(f"Early stopping at epoch {epoch} (no improvement for {args.early_stop}).")
        break

print(f"Best val MSE: {best_val:.6f} @ epoch {best_iter}")
print(f"Saved best:   {ckpt_best}")
print(f"Saved latest: {ckpt_last}")

# ---- shared results row ----
if write_result is not None:
    params = {
        "epochs": args.epochs,
        "early_stop": args.early_stop,
        "batch_size": args.batch_size,
        "learn_rate": args.learn_rate,
        "weight_decay": args.weight_decay,
        "function": args.function,
        "embedding_dim": args.embedding_dim,
        "latent_dim": args.latent_dim,
        "num_heads": args.num_heads,
        "depth": args.depth,
        "seed": args.seed,
        "observation_time": args.observation_time,
        "use_cycle": True,
        "cycle_length": args.cycle_length,
        "time_max_hours": args.time_max_hours,
    }
    metrics = {
        "best_epoch": best_iter,
        "val_mse_best": float(best_val),
        "val_rmse_best": float((best_val + 1e-8) ** 0.5),
        "train_loss_last_batch": last_train_loss,
        "test_mse_best": (float(test_report["mse"]) if test_report else None),
        "test_rmse_best": (float(test_report["rmse"]) if test_report else None),
    }
    write_result(
        model_name="IC-FLD-RCF",
        dataset=dataset_name,
        metrics=metrics,
        params=params,
        run_id=str(experiment_id),
    )

if writer:
    writer.flush(); writer.close()
