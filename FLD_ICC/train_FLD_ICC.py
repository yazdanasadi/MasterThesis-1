#!/usr/bin/env python
# coding: utf-8
"""
IC-FLD training with t-PatchGNN preprocessing (no patches).
- Supports: -d / -ot / -bs / --epochs / --early-stop / --lr / --wd / --gpu / --resume / --tbon / --logdir
- Model flags: -fn / -ed / -ld / -nh / --depth / --harmonics / --use-cycle / --cycle-length / --time-max-hours
"""

import argparse, sys, os, time, random, warnings, math, inspect, glob
from random import SystemRandom
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

# ---- project root / lib discovery ----
HERE = Path(__file__).resolve().parent
REPO = HERE if (HERE / "lib").exists() else HERE.parent
sys.path.append(str(REPO))
try:
    import lib.utils as utils
    from lib.parse_datasets import parse_datasets
except ModuleNotFoundError:
    sys.path.append(str(REPO / ".."))
    import lib.utils as utils
    from lib.parse_datasets import parse_datasets

# IC-FLD core (your inter-channel model)
from FLD_ICC import IC_FLD

# optional shared result writer
try:
    from write_result import write_result
except Exception:
    write_result = None

# ---------------- CLI ----------------
p = argparse.ArgumentParser(
    description="FLD-ICC training with t-PatchGNN preprocessing (no patches)"
)
p.add_argument("-d", "--dataset", type=str, default="physionet",
               choices=["physionet", "mimic", "ushcn", "activity"])
p.add_argument("-ot", "--observation-time", type=int, default=24,
               help="hours (ms for activity) as historical window")
p.add_argument("-bs", "--batch-size", type=int, default=32)
p.add_argument("-q", "--quantization", type=float, default=0.0)
p.add_argument("-n", type=int, default=int(1e8))

# model hyperparams
p.add_argument("-fn", "--function", type=str, default="L", choices=["C", "L", "Q", "S"])
p.add_argument("-ed", "--embedding-dim", type=int, default=64, help="total embedding dim")
p.add_argument("-nh", "--num-heads", type=int, default=2)
p.add_argument("-ld", "--latent-dim", type=int, default=64)
p.add_argument("--depth", type=int, default=2, help="decoder depth (layers)")
p.add_argument("--harmonics", type=int, default=2)

# residual cycle options
p.add_argument("--use-cycle", action="store_true")
p.add_argument("--cycle-length", type=int, default=24)
p.add_argument("--time-max-hours", type=int, default=48,
               help="un-normalized time max for cycle phases")

# training hyperparams
p.add_argument("--epochs", type=int, default=100)
p.add_argument("--early-stop", type=int, default=10)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--wd", type=float, default=0.0)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--gpu", type=str, default="0")
p.add_argument("--resume", type=str, default="", help='"" or "auto" or path to .pt')

# tensorboard
p.add_argument("--tbon", action="store_true")
p.add_argument("--logdir", type=str, default="runs")
p.add_argument("--tbgraph", action="store_true")

args = p.parse_args()

# ---------------- Setup ----------------
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
experiment_id = int(SystemRandom().random() * 10000000)

# TensorBoard
writer = None
if args.tbon:
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=args.logdir)

# checkpoints
SAVE_DIR = HERE / "saved_models"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
base = f"ICFLD{'-RCF' if args.use_cycle else ''}-{args.dataset}-{experiment_id}"

ckpt_best = SAVE_DIR / (base + ".best.pt")
ckpt_last = SAVE_DIR / (base + ".latest.pt")

# ---------------- Data via tPatchGNN preprocess (no patches) ----------------
dataset_map = {"physionet":"physionet","p12":"physionet",
               "mimic":"mimic","mimiciii":"mimic","ushcn":"ushcn","activity":"activity"}
ds_name = dataset_map.get(args.dataset.lower(), args.dataset)

pd_args = SimpleNamespace(
    state="def",
    n=args.n,
    hop=1, nhead=1, tf_layer=1, nlayer=1,
    epoch=args.epochs, patience=args.early_stop,
    history=int(args.observation_time),
    patch_size=8.0, stride=8.0,
    logmode="a",
    lr=args.lr, w_decay=args.wd,
    batch_size=int(args.batch_size),
    save="experiments/", load=None,
    seed=int(args.seed), dataset=ds_name,
    quantization=float(args.quantization),
    model="IC-FLD", outlayer="Linear",
    hid_dim=64, te_dim=10, node_dim=10,
    gpu=args.gpu,
)
pd_args.npatch = int(np.ceil((pd_args.history - pd_args.patch_size) / pd_args.stride)) + 1
pd_args.device = DEVICE

data_obj = parse_datasets(pd_args, patch_ts=False)
INPUT_DIM = data_obj["input_dim"]

# ---------------- Model ----------------
sig = inspect.signature(IC_FLD.__init__)
params = sig.parameters
has_varkw = any(p.kind == p.VAR_KEYWORD for p in params.values())

def _maybe(name: str) -> bool:
    # add the kwarg if explicitly present, or if the class accepts **kwargs
    return name in params or has_varkw

model_kwargs = {
    "input_dim": INPUT_DIM,
    "latent_dim": args.latent_dim,
    "num_heads": args.num_heads,
    "embed_dim": args.embedding_dim,
    "function": args.function,
    # "device": DEVICE,
}
if _maybe("depth"):          model_kwargs["depth"] = args.depth
if _maybe("harmonics"):      model_kwargs["harmonics"] = args.harmonics
if _maybe("use_cycle"):      model_kwargs["use_cycle"] = args.use_cycle
if _maybe("cycle_length"):   model_kwargs["cycle_length"] = args.cycle_length
if _maybe("time_max_hours"): model_kwargs["time_max_hours"] = args.time_max_hours

MODEL = IC_FLD(**model_kwargs).to(DEVICE)

# ---------------- Loss / utils ----------------
def mse_masked(y: torch.Tensor, yhat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.float()
    return (mask * (y - yhat) ** 2).sum() / mask.sum().clamp_min(1.0)

def batch_to_icfld(batch: dict, input_dim: int, device: torch.device):
    obs_tp = batch["observed_tp"].to(device)        # [B,L]
    obs_x  = batch["observed_data"].to(device)      # [B,L,D]
    obs_m  = batch["observed_mask"].to(device)      # [B,L,D]
    tp_pred = batch["tp_to_predict"].to(device)     # [B,Ty]
    y = batch["data_to_predict"].to(device)         # [B,Ty,D]
    y_mask = batch["mask_predicted_data"].to(device)# [B,Ty,D]
    if obs_x.dim() == 3 and obs_x.shape[1] == input_dim:
        obs_x = obs_x.transpose(1, 2).contiguous()
        obs_m = obs_m.transpose(1, 2).contiguous()
    return obs_tp, obs_x, obs_m, tp_pred, y, y_mask

# ---------------- Optimizer / scheduler ----------------
optimizer = optim.AdamW(MODEL.parameters(), lr=args.lr, weight_decay=args.wd)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=10, factor=0.5, verbose=False)

num_train_batches = data_obj["n_train_batches"]
best_val = float("inf"); best_iter = 0; test_report = None
last_train_loss = None  

print("PID, device:", os.getpid(), DEVICE)
print(f"Dataset={ds_name}, INPUT_DIM={INPUT_DIM}, history={pd_args.history}")
print("n_train_batches:", num_train_batches)

# ---------------- Resume++ ----------------
start_epoch = 1
if args.resume:
    def _state_dict_from(ck):
        # Accept either raw state_dict or our dict with metadata
        return ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck

    def _is_compatible_ckpt(ck_args: dict) -> bool:
        # Only accept checkpoints that match key hyperparams of the current run
        # Fallbacks handle older checkpoints that may not have all fields saved.
        if not isinstance(ck_args, dict):
            return False
        return (
            ck_args.get("dataset", ds_name) == ds_name and
            int(ck_args.get("embedding_dim", args.embedding_dim)) == int(args.embedding_dim) and
            int(ck_args.get("num_heads", args.num_heads)) == int(args.num_heads) and
            int(ck_args.get("latent_dim", args.latent_dim)) == int(args.latent_dim) and
            int(ck_args.get("observation_time", args.observation_time)) == int(args.observation_time) and
            bool(ck_args.get("use_cycle", args.use_cycle)) == bool(args.use_cycle)
        )

    def _filter_compatible(sd_model: dict, sd_file: dict):
        # Keep only keys whose shapes match the current model
        filtered = {}
        skipped = []
        for k, v in sd_file.items():
            if k in sd_model and sd_model[k].shape == v.shape:
                filtered[k] = v
            else:
                skipped.append(k)
        return filtered, skipped

    if args.resume == "auto":
        load_path = ""
        # Find most recent compatible *.latest.pt
        candidates = sorted(SAVE_DIR.glob("ICFLD*.*.pt"), key=os.path.getmtime)
        for pth in reversed(candidates):
            try:
                ck = torch.load(pth, map_location="cpu")
            except Exception:
                continue
            ck_args = ck.get("args", {})
            if _is_compatible_ckpt(ck_args):
                load_path = str(pth)
                break
        if not load_path:
            print("[resume] no compatible checkpoint found for --resume auto; starting fresh.")
    else:
        load_path = args.resume

    if load_path and Path(load_path).exists():
        ckpt = torch.load(load_path, map_location="cpu")
        raw_sd = _state_dict_from(ckpt)
        model_sd = MODEL.state_dict()
        filt_sd, skipped = _filter_compatible(model_sd, raw_sd)

        missing, unexpected = MODEL.load_state_dict(filt_sd, strict=False)
        print(f"[resume] loaded from {load_path}")
        if skipped:
            print(f"[resume] skipped {len(skipped)} incompatible keys (shape mismatch).")
        if missing:
            print(f"[resume] model missing {len(missing)} keys (left at init).")
        if unexpected:
            print(f"[resume] unexpected {len(unexpected)} keys in checkpoint.")

        # Restore optimizer/epoch/best if present and shapes were largely compatible
        if isinstance(ckpt, dict):
            if "optimizer" in ckpt:
                try:
                    optimizer.load_state_dict(ckpt["optimizer"])
                except Exception:
                    print("[resume] optimizer state not loaded (probably param mismatch).")
            if "epoch" in ckpt:
                start_epoch = int(ckpt["epoch"]) + 1
            if "best_val" in ckpt:
                best_val = float(ckpt["best_val"])
            if "best_iter" in ckpt:
                best_iter = int(ckpt["best_iter"])
            print(f"[resume] start_epoch={start_epoch}, best_val={best_val:.6f}, best_iter={best_iter}")
    elif args.resume:
        print(f"[resume] path not found: {load_path} (starting fresh)")

# ---------------- Eval helper ----------------
def evaluate(loader, nb):
    total = 0.0; cnt = 0.0
    for _ in range(nb):
        b = utils.get_next_batch(loader)
        T, X, M, TY, Y, YM = batch_to_icfld(b, INPUT_DIM, DEVICE)
        with torch.no_grad():
            YH = MODEL(T, X, M, TY, denorm_time_max=(args.time_max_hours if args.use_cycle else None))
        total += float((YM * (Y - YH) ** 2).sum().item())
        cnt   += float(YM.sum().item())
    mse = total / max(1.0, cnt)
    rmse = (mse + 1e-8) ** 0.5
    return {"loss": mse, "mse": mse, "rmse": rmse}

# ---------------- Train loop ----------------
for epoch in range(start_epoch, args.epochs + 1):
    st = time.time()
    MODEL.train()
    for _ in range(num_train_batches):
        optimizer.zero_grad(set_to_none=True)
        batch = utils.get_next_batch(data_obj["train_dataloader"])
        T, X, M, TY, Y, YM = batch_to_icfld(batch, INPUT_DIM, DEVICE)
        YH = MODEL(T, X, M, TY, denorm_time_max=(args.time_max_hours if args.use_cycle else None))
        loss = mse_masked(Y, YH, YM)
        loss.backward()
        optimizer.step()
        last_train_loss = float(loss.item())
    MODEL.eval()
    val_res  = evaluate(data_obj["val_dataloader"], data_obj["n_val_batches"])

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

    dt = time.time() - st
    print(f"- Epoch {epoch:03d} | train_loss(one-batch): {loss.item():.6f} | "
          f"val_mse: {val_res['mse']:.6f} | val_rmse: {val_res['rmse']:.6f} | "
          + (f"best@{best_iter} test_mse: {test_report['mse']:.6f} rmse: {test_report['rmse']:.6f} | " if test_report else "")
          + f"time: {dt:.2f}s")

    if args.tbon and writer:
        if last_train_loss is not None:
            writer.add_scalar("loss/train_last_batch", last_train_loss, epoch)
        writer.add_scalar("val/mse", float(val_res["mse"]), epoch)
        writer.add_scalar("val/rmse", float(val_res["rmse"]), epoch)
        if args.tbgraph and epoch == 1:
            try:
                dummy_b = utils.get_next_batch(data_obj["train_dataloader"])
                T, X, M, TY, Y, YM = batch_to_icfld(dummy_b, INPUT_DIM, DEVICE)
                writer.add_graph(MODEL, (T, X, M, TY))
            except Exception as e:
                print(f"[tbgraph] skipped: {e}")

    if (epoch - best_iter) >= args.early_stop:
        print(f"Early stopping at epoch {epoch} (no improvement for {args.early_stop} epochs).")
        break
# If we resumed beyond max epoch, no training step ran and last_train_loss is None.
    if last_train_loss is None:
        try:
            b = utils.get_next_batch(data_obj["train_dataloader"])
            T, X, M, TY, Y, YM = batch_to_icfld(b, INPUT_DIM, DEVICE)
            with torch.no_grad():
                YH = MODEL(T, X, M, TY, denorm_time_max=(args.time_max_hours if args.use_cycle else None))
            last_train_loss = float(mse_masked(Y, YH, YM).item())
        except Exception:
            last_train_loss = None  # still fine; we’ll write None to CSV
    print(f"Best val MSE: {best_val:.6f} @ epoch {best_iter}")
    print(f"Saved best: {ckpt_best}")
    print(f"Saved latest: {ckpt_last}")


if write_result is not None:
    params = {
        "epochs": args.epochs,
        "early_stop": args.early_stop,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "wd": args.wd,
        "function": args.function,
        "embedding_dim": args.embedding_dim,
        "latent_dim": args.latent_dim,
        "num_heads": args.num_heads,
        "depth": args.depth,
        "harmonics": args.harmonics,
        "use_cycle": args.use_cycle,
        "cycle_length": args.cycle_length,
        "time_max_hours": args.time_max_hours,
        "seed": args.seed,
        "observation_time": args.observation_time,
    }
    metrics = {
         "best_epoch": best_iter,
        "val_mse_best": float(best_val),
        "val_rmse_best": float((best_val + 1e-8) ** 0.5),
        "train_loss_last_batch": float(loss.item()),
        "test_mse_best": (float(test_report["mse"]) if test_report else None),
        "test_rmse_best": (float(test_report["rmse"]) if test_report else None),
    }
    write_result(
        model_name="IC-FLD" + ("-RCF" if args.use_cycle else ""),
        dataset=ds_name,
        metrics=metrics,
        params=params,
        run_id=str(experiment_id),
    )

if writer:
    writer.flush()
    writer.close()