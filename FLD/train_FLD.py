#!/usr/bin/env python
# coding: utf-8

import argparse, sys, os, time, random, warnings
from random import SystemRandom
from types import SimpleNamespace
from pathlib import Path
from write_result import write_result

import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# ---- tPatchGNN libs ----
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

from FLD import FLD  # your FLD model file (unchanged core)

# --------- CLI ---------
parser = argparse.ArgumentParser(description="FLD training with t-PatchGNN preprocessing (no patches)")
parser.add_argument("-r", "--run_id", default=None, type=str)
parser.add_argument("-e", "--epochs", default=300, type=int)
parser.add_argument("-es", "--early-stop", default=30, type=int)
parser.add_argument("-bs", "--batch-size", default=64, type=int)
parser.add_argument("-lr", "--learn-rate", default=1e-3, type=float)
parser.add_argument("-wd", "--weight-decay", default=0.0, type=float)
parser.add_argument("-s", "--seed", default=0, type=int)
parser.add_argument("-d", "--dataset", default="ushcn", type=str, help="physionet | mimic | ushcn | activity")
parser.add_argument("-ot", "--observation-time", default=24, type=int, help="history window length")
parser.add_argument("-fn", "--function", default="C", choices=("L", "S", "C", "Q"))
parser.add_argument("-ed", "--embedding-dim", default=4, type=int)
parser.add_argument("-nh", "--num-heads", default=2, type=int)
parser.add_argument("-dp", "--depth", default=1, type=int)
parser.add_argument("--gpu", default="0", type=str)
parser.add_argument("--resume", default="", type=str, help="'auto' or path to a .pt checkpoint")
# TensorBoard
parser.add_argument("--tbon", action="store_true", help="Enable TensorBoard logging")
parser.add_argument("--logdir", type=str, default="runs", help="TensorBoard log root")
args = parser.parse_args()

# --------- Setup ---------
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)

experiment_id = int(SystemRandom().random() * 10000000)
Path("saved_models").mkdir(parents=True, exist_ok=True)
model_best_path = f"saved_models/FLD-{args.function}_{args.dataset}_{experiment_id}.best.pt"
model_latest_path = f"saved_models/FLD-{args.function}_{args.dataset}_{experiment_id}.latest.pt"

# --------- Build loaders via tPatchGNN (no patches) ---------
dataset_map = {"physionet":"physionet","p12":"physionet","mimic":"mimic","mimiciii":"mimic","ushcn":"ushcn","activity":"activity"}
dataset_name = dataset_map.get(args.dataset.lower(), args.dataset)

pd_args = SimpleNamespace(
    state="def", n=int(1e8), hop=1, nhead=1, tf_layer=1, nlayer=1,
    epoch=args.epochs, patience=args.early_stop, history=int(args.observation_time),
    patch_size=8.0, stride=8.0, logmode="a",
    lr=args.learn-rate if hasattr(args, "learn-rate") else args.learn_rate,
    w_decay=args.weight_decay, batch_size=int(args.batch_size),
    save="experiments/", load=None, seed=int(args.seed),
    dataset=dataset_name, quantization=0.0, model="FLD",
    outlayer="Linear", hid_dim=64, te_dim=10, node_dim=10, gpu=args.gpu,
)
pd_args.npatch = int(np.ceil((pd_args.history - pd_args.patch_size) / pd_args.stride)) + 1
pd_args.device = DEVICE

data_obj = parse_datasets(pd_args, patch_ts=False)
INPUT_DIM = data_obj["input_dim"]
num_train_batches = data_obj["n_train_batches"]

print("PID, device:", os.getpid(), DEVICE)
print(f"Dataset={dataset_name}, INPUT_DIM={INPUT_DIM}, history={pd_args.history}")
print("n_train_batches:", num_train_batches)

# --------- Build model ---------
MODEL = FLD(
    input_dim=INPUT_DIM,
    latent_dim=20,
    embed_dim_per_head=args.embedding_dim,
    num_heads=args.num_heads,
    function=args.function,
    depth=args.depth,
    device=DEVICE,
).to(DEVICE)

# --------- Loss/metrics helpers ---------
def _orient_time_last(x: torch.Tensor, input_dim: int) -> torch.Tensor:
    if x.dim() != 3: raise ValueError(f"Expected 3D tensor, got {x.shape}")
    if x.shape[-1] == input_dim: return x
    if x.shape[1] == input_dim:  return x.transpose(1, 2).contiguous()
    raise ValueError(f"Cannot infer feature axis from {x.shape} with D={input_dim}")

def mse_masked(y: torch.Tensor, yhat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.float()
    return (mask * (y - yhat) ** 2).sum() / mask.sum().clamp_min(1.0)

@torch.no_grad()
def evaluate(model, loader, nbatches, input_dim, device):
    total = 0.0; cnt = 0.0
    for _ in range(nbatches):
        b = utils.get_next_batch(loader)
        T   = b["observed_tp"].to(device)                                  # [B,L]
        X   = _orient_time_last(b["observed_data"].to(device), input_dim)  # [B,L,D]
        M   = _orient_time_last(b["observed_mask"].to(device), input_dim)  # [B,L,D]
        TY  = b["tp_to_predict"].to(device)                                # [B,Ty]
        Y   = _orient_time_last(b["data_to_predict"].to(device), input_dim)# [B,Ty,D]
        YM  = _orient_time_last(b.get("mask_predicted_data",
                               torch.ones_like(Y, dtype=torch.float32)).to(device), input_dim)
        YH = model(T, X, M, TY)
        total += float(((Y - YH) ** 2 * YM).sum().item()); cnt += float(YM.sum().item())
    mse = total / max(1.0, cnt); rmse = (mse + 1e-8) ** 0.5
    return {"loss": mse, "mse": mse, "rmse": rmse}

# --------- Optim / sched ---------
optimizer = optim.AdamW(MODEL.parameters(),
                        lr=args.learn_rate if hasattr(args,"learn_rate") else args.__dict__.get("learn-rate",1e-3),
                        weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=10, factor=0.5, verbose=True)

best_val = float("inf"); best_iter = 0; test_report = None

# --------- TensorBoard (optional) ---------
writer = None
if args.tbon:
    run_name = f"FLD_{dataset_name}_{int(time.time())}"
    writer = SummaryWriter(log_dir=os.path.join(args.logdir, run_name))
    try:
        b = utils.get_next_batch(data_obj["train_dataloader"])
        T  = b["observed_tp"].to(DEVICE)
        X  = _orient_time_last(b["observed_data"].to(DEVICE), INPUT_DIM)
        M  = _orient_time_last(b["observed_mask"].to(DEVICE), INPUT_DIM)
        TY = b["tp_to_predict"].to(DEVICE)
        writer.add_graph(MODEL, (T, X, M, TY))
    except Exception:
        pass

# --------- Train ---------
try:
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        MODEL.train()
        for _ in range(num_train_batches):
            optimizer.zero_grad(set_to_none=True)
            b = utils.get_next_batch(data_obj["train_dataloader"])
            T  = b["observed_tp"].to(DEVICE)
            X  = _orient_time_last(b["observed_data"].to(DEVICE), INPUT_DIM)
            M  = _orient_time_last(b["observed_mask"].to(DEVICE), INPUT_DIM)
            TY = b["tp_to_predict"].to(DEVICE)
            Y  = _orient_time_last(b["data_to_predict"].to(DEVICE), INPUT_DIM)
            YM = _orient_time_last(b.get("mask_predicted_data",
                                  torch.ones_like(Y, dtype=torch.float32)).to(DEVICE), INPUT_DIM)
            YH = MODEL(T, X, M, TY)
            loss = mse_masked(Y, YH, YM)
            loss.backward(); optimizer.step()

        MODEL.eval()
        with torch.no_grad():
            val_res = evaluate(MODEL, data_obj["val_dataloader"], data_obj["n_val_batches"], INPUT_DIM, DEVICE)
            if val_res["mse"] < best_val:
                best_val = val_res["mse"]; best_iter = epoch
                test_report = evaluate(MODEL, data_obj["test_dataloader"], data_obj["n_test_batches"], INPUT_DIM, DEVICE)
                torch.save({"state_dict": MODEL.state_dict(), "args": vars(args), "input_dim": INPUT_DIM}, model_best_path)

        scheduler.step(val_res["loss"])

        dt = time.time() - t0
        print(
            f"- Epoch {epoch:03d} | train_loss(one-batch): {loss.item():.6f} | "
            f"val_mse: {val_res['mse']:.6f} | val_rmse: {val_res['rmse']:.6f} | "
            + (f"best@{best_iter} test_mse: {test_report['mse']:.6f} rmse: {test_report['rmse']:.6f} | " if test_report else "")
            + f"time: {dt:.2f}s"
        )

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

        # latest (per-epoch)
        torch.save({"state_dict": MODEL.state_dict(), "args": vars(args), "input_dim": INPUT_DIM}, model_latest_path)

except KeyboardInterrupt:
    print("\n[interrupt] KeyboardInterrupt — saving latest and exiting.")
    torch.save({"state_dict": MODEL.state_dict(), "args": vars(args), "input_dim": INPUT_DIM}, model_latest_path)
    raise

print(f"Best val MSE: {best_val:.6f} @ epoch {best_iter}")
print(f"Saved best:   {model_best_path}")
print(f"Saved latest: {model_latest_path}")
# ---- write shared results row ----
params = {
    "epochs": args.epochs,
    "early_stop": args.early_stop,
    "batch_size": args.batch_size,
    "learn_rate": (args.learn_rate if hasattr(args, "learn_rate")
                   else args.__dict__.get("learn-rate", None)),
    "weight_decay": args.weight_decay,
    "function": args.function,
    "embedding_dim_per_head": args.embedding_dim,
    "num_heads": args.num_heads,
    "depth": args.depth,
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
    model_name="FLD",
    dataset=dataset_name,
    metrics=metrics,
    params=params,
    run_id=str(experiment_id),
)

if writer:
    writer.flush(); writer.close()
