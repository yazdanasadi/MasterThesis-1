#!/usr/bin/env python
# coding: utf-8

import argparse, sys, os, time, random, warnings, glob
from random import SystemRandom
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from write_result import write_result

# ----------------- Resolve repo root & imports -----------------
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(str(THIS_DIR))

try:
    import lib.utils as utils
    from lib.parse_datasets import parse_datasets
except ModuleNotFoundError as e:
    raise RuntimeError(f"Could not import repo packages from {REPO_ROOT}. Run from the repo root. Original error: {e}")

from gratif import GrATiF
from fullgraph import FullGraph
from bipartitegraph import BipartiteGraph

# ----------------- CLI -----------------
p = argparse.ArgumentParser("GraFITi with t-PatchGNN preprocessing (forecasting)")
p.add_argument("-d",  "--dataset",           default="physionet", choices=("physionet","mimic","ushcn","activity"))
p.add_argument("-ot", "--observation-time",  default=24,  type=int)
p.add_argument("-bs", "--batch-size",        default=32,  type=int)
p.add_argument("-q",  "--quantization",      default=0.0, type=float)
p.add_argument("--epochs",       default=100, type=int)
p.add_argument("--early-stop",   default=15,  type=int)
p.add_argument("--lr",           default=1e-3,type=float)
p.add_argument("--wd",           default=0.0, type=float)
p.add_argument("--seed",         default=0,   type=int)
p.add_argument("--gpu",          default="0", type=str)
p.add_argument("--resume",       default="",  type=str, help="'auto' or path to .pt")
p.add_argument("--encoder",      default="gratif", choices=("gratif","fullgraph","bipartite"))
# TensorBoard
p.add_argument("--tbon", action="store_true", help="Enable TensorBoard logging")
p.add_argument("--logdir", type=str, default="runs", help="TensorBoard log root")
args = p.parse_args()

# ----------------- Setup -----------------
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)

dataset_alias = {"physionet":"physionet","p12":"physionet",
                 "mimic":"mimic","mimiciii":"mimic",
                 "ushcn":"ushcn","activity":"activity"}
ds_name = dataset_alias.get(args.dataset.lower(), args.dataset)

pd_args = SimpleNamespace(
    state="def", n=int(1e8), hop=1, nhead=1, tf_layer=1, nlayer=1,
    epoch=args.epochs, patience=args.early_stop, history=int(args.observation_time),
    patch_size=8.0, stride=8.0, logmode="a", lr=args.lr, w_decay=args.wd,
    batch_size=int(args.batch_size), save=str(REPO_ROOT / "experiments/"), load=None,
    seed=int(args.seed), dataset=ds_name, quantization=float(args.quantization),
    model="GraFITi", outlayer="Linear", hid_dim=64, te_dim=10, node_dim=10, gpu=args.gpu,
)
pd_args.npatch = int(np.ceil((pd_args.history - pd_args.patch_size) / pd_args.stride)) + 1
pd_args.device = DEVICE

data_obj = parse_datasets(pd_args, patch_ts=False)
INPUT_DIM = data_obj["input_dim"]
num_train_batches = data_obj["n_train_batches"]

print("PID, device:", os.getpid(), DEVICE)
print(f"Dataset={ds_name}, INPUT_DIM={INPUT_DIM}, history={pd_args.history}")
print("n_train_batches:", num_train_batches)

# ----------------- Model -----------------
if args.encoder == "gratif":
    EncoderModel = GrATiF
elif args.encoder == "fullgraph":
    EncoderModel = FullGraph
else:
    EncoderModel = BipartiteGraph

MODEL = EncoderModel(
    input_dim=INPUT_DIM,
    attn_head=4,
    latent_dim=128,
    n_layers=2,
    device=DEVICE,
).to(DEVICE)

optimizer = optim.AdamW(MODEL.parameters(), lr=args.lr, weight_decay=args.wd)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=10, factor=0.5, verbose=True)

def _orient_time_last(x, input_dim):
    if x.dim()!=3: raise ValueError(f"Expected 3D, got {x.shape}")
    if x.shape[-1]==input_dim: return x
    if x.shape[1]==input_dim:  return x.transpose(1,2).contiguous()
    raise ValueError(f"Bad shape {x.shape} for D={input_dim}")

def predict_fn(model, batch):
    T   = batch["observed_tp"].to(DEVICE)                                   # [B,L]
    X   = _orient_time_last(batch["observed_data"].to(DEVICE), INPUT_DIM)   # [B,L,D]
    M   = _orient_time_last(batch["observed_mask"].to(DEVICE), INPUT_DIM)   # [B,L,D]
    TY  = batch["tp_to_predict"].to(DEVICE)                                 # [B,Ty]
    Y   = _orient_time_last(batch["data_to_predict"].to(DEVICE), INPUT_DIM) # [B,Ty,D]
    YM  = _orient_time_last(batch.get("mask_predicted_data",
                         torch.ones_like(Y, dtype=torch.float32)).to(DEVICE), INPUT_DIM)

    # Align time dims for mask-add line inside model if needed
    Lc, Lt = X.shape[1], Y.shape[1]
    if Lt != Lc:
        if Lt < Lc:
            pad_t = Lc - Lt
            pad_vals = torch.zeros(Y.size(0), pad_t, Y.size(2), device=Y.device, dtype=Y.dtype)
            pad_mask = torch.zeros_like(pad_vals)
            Y  = torch.cat([Y,  pad_vals], dim=1)
            YM = torch.cat([YM, pad_mask], dim=1)
        else:
            Y  = Y[:, :Lc]
            YM = YM[:, :Lc]

    out, tgt_vals, tgt_mask = model(T, X, M, TY, Y, YM)      # model returns (pred, tgt, mask)
    return tgt_vals, out.squeeze(-1), tgt_mask

@torch.no_grad()
def evaluate(model, loader, nbatches):
    total = 0.0; denom = 0.0
    for _ in range(nbatches):
        b = utils.get_next_batch(loader)
        Y, YH, MASK = predict_fn(model, b)
        total += float(((Y - YH)**2 * MASK).sum().item())
        denom += float(MASK.sum().item())
    mse = total / max(1.0, denom); rmse = (mse + 1e-8) ** 0.5
    return {"loss": mse, "mse": mse, "rmse": rmse}

# ----------------- Resume -----------------
CKPT_DIR = REPO_ROOT / "saved_models"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
base_name = f"GraFITi-{args.encoder}_{ds_name}"
experiment_id = int(SystemRandom().random() * 10_000_000)
best_path   = CKPT_DIR / f"{base_name}_{experiment_id}.best.pt"
latest_path = CKPT_DIR / f"{base_name}_{experiment_id}.latest.pt"

start_epoch = 1; resume_batch_idx = 0
best_val = float("inf"); best_iter = 0; test_report = None

# ----------------- TensorBoard (optional) -----------------
writer = None
if args.tbon:
    run_name = f"GraFITi_{args.encoder}_{ds_name}_{int(time.time())}"
    writer = SummaryWriter(log_dir=os.path.join(args.logdir, run_name))

# ----------------- Train -----------------
try:
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        MODEL.train()

        for bidx in range(num_train_batches):
            batch = utils.get_next_batch(data_obj["train_dataloader"])
            Y, YH, MASK = predict_fn(MODEL, batch)
            loss = ((Y - YH)**2 * MASK).sum() / MASK.sum().clamp_min(1.0)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            torch.save({
                "model": MODEL.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "batch_idx": bidx,
                "best_val": best_val,
                "best_iter": best_iter,
                "test_report": test_report,
                "args": vars(args),
                "input_dim": INPUT_DIM,
                "experiment_id": experiment_id,
                "dataset": ds_name,
                "encoder": args.encoder,
            }, latest_path)

        val_res = evaluate(MODEL, data_obj["val_dataloader"], data_obj["n_val_batches"])
        if val_res["mse"] < best_val:
            best_val = val_res["mse"]; best_iter = epoch
            test_report = evaluate(MODEL, data_obj["test_dataloader"], data_obj["n_test_batches"])
            torch.save({
                "state_dict": MODEL.state_dict(),
                "args": vars(args),
                "input_dim": INPUT_DIM,
                "experiment_id": experiment_id,
                "dataset": ds_name,
                "encoder": args.encoder,
                "best_val": best_val,
                "best_iter": best_iter,
            }, best_path)

        scheduler.step(val_res["loss"])
        dt = time.time() - t0
        print(f"- Epoch {epoch:03d} | train_loss(one-batch): {loss.item():.6f} | "
              f"val_mse: {val_res['mse']:.6f} | val_rmse: {val_res['rmse']:.6f} | "
              + (f"best@{best_iter} test_mse: {test_report['mse']:.6f} rmse: {test_report['rmse']:.6f} | " if test_report else "")
              + f"time: {dt:.2f}s")

        if writer:
            writer.add_scalar("train/loss_one_batch", float(loss.item()), epoch)
            writer.add_scalar("val/mse",  float(val_res["mse"]),  epoch)
            writer.add_scalar("val/rmse", float(val_res["rmse"]), epoch)
            if test_report:
                writer.add_scalar("test/mse_best",  float(test_report["mse"]),  epoch)
                writer.add_scalar("test/rmse_best", float(test_report["rmse"]), epoch)

        if (epoch - best_iter) >= args.early_stop:
            print(f"Early stopping at epoch {epoch} (no improvement for {args.early_stop}).")
            break

except KeyboardInterrupt:
    print("\n[interrupt] Caught KeyboardInterrupt; saving latest checkpoint...")
    try:
        ckpt = torch.load(latest_path, map_location=DEVICE) if latest_path.exists() else None
        if not ckpt:
            torch.save({"state_dict": MODEL.state_dict(), "args": vars(args), "input_dim": INPUT_DIM}, latest_path)
    except Exception as e:
        print(f"[interrupt] failed to save latest: {e}")
    raise

print(f"Saved best:   {best_path}")
print(f"Saved latest: {latest_path}")

params = {
    "epochs": args.epochs,
    "early_stop": args.early_stop,
    "batch_size": args.batch_size,
    "lr": args.lr,
    "wd": args.wd,
    "encoder": args.encoder,
    # model dims used in this trainer:
    "attn_head": 4,
    "latent_dim": 128,
    "n_layers": 2,
    "seed": args.seed,
    "observation_time": args.observation_time,
}
metrics = {
    "best_epoch": best_iter if "best_iter" in locals() else None,
    "val_mse_best": best_val,
    "val_rmse_best": float((best_val + 1e-8) ** 0.5),
    "train_loss_last_batch": float(loss.item()),
    "test_mse_best": (float(test_report["mse"]) if test_report else None),
    "test_rmse_best": (float(test_report["rmse"]) if test_report else None),
}
write_result(
    model_name=f"GraFITi-{args.encoder}",
    dataset=ds_name,
    metrics=metrics,
    params=params,
    run_id=str(experiment_id),
)

if writer:
    writer.flush(); writer.close()
