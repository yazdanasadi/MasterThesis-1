#!/usr/bin/env python
# coding: utf-8

import argparse, sys, os, time, random, warnings, glob
from pathlib import Path
from types import SimpleNamespace
from random import SystemRandom
from write_result import write_result

import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# ---- repo paths ----
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---- libs ----
try:
    import lib.utils as tp_utils
    from lib.parse_datasets import parse_datasets
except ModuleNotFoundError:
    sys.path.append(str(REPO_ROOT))
    import lib.utils as tp_utils
    from lib.parse_datasets import parse_datasets

# your model defs (unchanged)
from models import enc_mtan_rnn, dec_mtan_rnn

# ---- CLI ----
p = argparse.ArgumentParser("mTAN training with t-PatchGNN preprocessing (forecasting)")
p.add_argument("-d", "--dataset", default="physionet", choices=("physionet","mimic","ushcn","activity"))
p.add_argument("-ot","--observation-time", default=24, type=int)
p.add_argument("-bs","--batch-size", default=32, type=int)
p.add_argument("--niters", default=100, type=int)
p.add_argument("--early-stop", default=10, type=int)
p.add_argument("--lr", default=1e-3, type=float)

# model dims
p.add_argument("--latent-dim", default=16, type=int)
p.add_argument("--rec-hidden", default=32, type=int)
p.add_argument("--gen-hidden", default=50, type=int)
p.add_argument("--embed-time", default=128, type=int)
p.add_argument("--num-heads", default=1, type=int)
p.add_argument("--k-iwae", default=10, type=int)
p.add_argument("--std", default=0.1, type=float)

# misc
p.add_argument("--seed", default=0, type=int)
p.add_argument("--gpu", default="0", type=str)
p.add_argument("--resume", default="", type=str, help="'auto' or path to .pt checkpoint")
# TensorBoard
p.add_argument("--tbon", action="store_true", help="Enable TensorBoard logging")
p.add_argument("--logdir", type=str, default="runs", help="TensorBoard log root")
args = p.parse_args()

# ---- setup ----
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)

dataset_alias = {"physionet":"physionet","p12":"physionet","mimic":"mimic","mimiciii":"mimic","ushcn":"ushcn","activity":"activity"}
ds_name = dataset_alias.get(args.dataset.lower(), args.dataset)

pd_args = SimpleNamespace(
    state="def", n=int(1e8), hop=1, nhead=1, tf_layer=1, nlayer=1,
    epoch=args.niters, patience=args.early_stop, history=int(args.observation_time),
    patch_size=8.0, stride=8.0, logmode="a", lr=args.lr, w_decay=0.0,
    batch_size=int(args.batch_size), save=str(REPO_ROOT / "experiments/"), load=None,
    seed=int(args.seed), dataset=ds_name, quantization=0.0, model="mTAN",
    outlayer="Linear", hid_dim=64, te_dim=10, node_dim=10, gpu=args.gpu, device=DEVICE,
)
pd_args.npatch = 1

data_obj = parse_datasets(pd_args, patch_ts=False)
INPUT_DIM = data_obj["input_dim"]
num_train_batches = data_obj["n_train_batches"]

print("PID, device:", os.getpid(), DEVICE)
print(f"Dataset={ds_name}, INPUT_DIM={INPUT_DIM}, history={pd_args.history}")
print("n_train_batches:", num_train_batches)

# ---- model ----
grid = torch.linspace(0, 1.0, args.embed_time).to(DEVICE)
rec = enc_mtan_rnn(INPUT_DIM, grid, latent_dim=args.latent_dim, nhidden=args.rec_hidden,
                   embed_time=args.embed_time, num_heads=args.num_heads, learn_emb=False, device=DEVICE).to(DEVICE)
dec = dec_mtan_rnn(INPUT_DIM, grid, latent_dim=args.latent_dim, nhidden=args.gen_hidden,
                   embed_time=args.embed_time, num_heads=args.num_heads, learn_emb=False, device=DEVICE).to(DEVICE)

optimizer = optim.AdamW(list(rec.parameters()) + list(dec.parameters()), lr=args.lr, weight_decay=0.0)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=10, factor=0.5, verbose=True)

def _orient_time_last(x):
    return x if x.shape[-1]==INPUT_DIM else x.transpose(1,2).contiguous()

def gaussian_loglik(y_true, y_pred_k, mask, sigma: float):
    K = y_pred_k.shape[0]
    y_true_k = y_true.unsqueeze(0).expand(K, -1, -1, -1)
    mask_k   = mask.unsqueeze(0).expand(K, -1, -1, -1).float()
    diff2 = ((y_true_k - y_pred_k) ** 2) * mask_k
    n_obs = mask_k.sum(dim=(2, 3))
    const = (n_obs > 0).float() * n_obs * np.log(2 * np.pi * (sigma ** 2))
    ll = -0.5 * (diff2.sum(dim=(2, 3)) / (sigma ** 2) + const)
    return ll  # [K,B]

def kl_q_standard_normal(qmu, qlv):
    v = torch.exp(qlv)
    return 0.5 * ((qmu ** 2 + v - 1.0 - qlv).sum(dim=(1, 2)))

def iw_elbo_loss(logpx, kl, K):
    # logpx: [K,B], kl: [B]
    # IWAE (one-sample/avg)
    log_w = logpx - kl.unsqueeze(0)   # [K,B]
    m = torch.max(log_w, dim=0, keepdim=True)[0]
    w = torch.exp(log_w - m)
    elbo = (m + torch.log(w.mean(dim=0) + 1e-12)).mean()
    return -elbo

@torch.no_grad()
def evaluate_mse(rec, dec, loader, nbatches, input_dim, device, num_sample=1):
    mse, denom = 0.0, 0.0
    for _ in range(nbatches):
        b = tp_utils.get_next_batch(loader)
        T_enc  = b["observed_tp"].to(device)
        X_enc  = _orient_time_last(b["observed_data"].to(device))
        M_enc  = _orient_time_last(b["observed_mask"].to(device))
        T_dec  = b["tp_to_predict"].to(device)
        Y_true = _orient_time_last(b["data_to_predict"].to(device))
        Y_mask = _orient_time_last(b.get("mask_predicted_data", torch.ones_like(Y_true)).to(device))
        out = rec(torch.cat((X_enc, M_enc), 2), T_enc)
        qmu, qlv = out[:, :, :args.latent_dim], out[:, :, args.latent_dim:]
        eps = torch.randn(num_sample, qmu.shape[0], qmu.shape[1], qmu.shape[2], device=device)
        z0  = eps * torch.exp(0.5 * qlv) + qmu
        z0  = z0.view(-1, qmu.shape[1], qmu.shape[2])
        Ty  = T_dec[None, :, :].repeat(num_sample, 1, 1).view(-1, T_dec.shape[1])
        pred = dec(z0, Ty).view(num_sample, X_enc.shape[0], Ty.shape[1], X_enc.shape[2]).mean(0)
        mse += float(((Y_true - pred) ** 2 * Y_mask).sum().item())
        denom += float(Y_mask.sum().item())
    mse = mse / max(1.0, denom); rmse = (mse + 1e-8) ** 0.5
    return {"mse": mse, "rmse": rmse, "loss": mse}

# ---- TensorBoard (optional) ----
writer = None
if args.tbon:
    run_name = f"mTAN_{ds_name}_{int(time.time())}"
    writer = SummaryWriter(log_dir=os.path.join(args.logdir, run_name))

best_val = float("inf"); best_iter = 0; test_report = None

# ---- Train ----
try:
    for it in range(1, args.niters + 1):
        t0 = time.time()
        rec.train(); dec.train()
        # one pass through the epoch
        for _ in range(num_train_batches):
            optimizer.zero_grad(set_to_none=True)
            b = tp_utils.get_next_batch(data_obj["train_dataloader"])
            T_enc  = b["observed_tp"].to(DEVICE)
            X_enc  = _orient_time_last(b["observed_data"].to(DEVICE))
            M_enc  = _orient_time_last(b["observed_mask"].to(DEVICE))
            T_dec  = b["tp_to_predict"].to(DEVICE)
            Y_true = _orient_time_last(b["data_to_predict"].to(DEVICE))
            Y_mask = _orient_time_last(b.get("mask_predicted_data",
                                     torch.ones_like(Y_true, dtype=torch.float32)).to(DEVICE))
            out = rec(torch.cat((X_enc, M_enc), 2), T_enc)
            qmu, qlv = out[:, :, :args.latent_dim], out[:, :, args.latent_dim:]
            K = args.k_iwae
            eps = torch.randn(K, qmu.shape[0], qmu.shape[1], qmu.shape[2], device=DEVICE)
            z0  = eps * torch.exp(0.5 * qlv) + qmu               # [K,B,L,Lat]
            z0  = z0.view(-1, qmu.shape[1], qmu.shape[2])        # [K*B,L,Lat]
            Ty  = T_dec[None, :, :].repeat(K, 1, 1).view(-1, T_dec.shape[1])
            pred = dec(z0, Ty)                                    # [K*B, Ty, D]
            pred = pred.view(K, X_enc.shape[0], pred.shape[1], pred.shape[2])  # [K,B,Ty,D]
            logpx = gaussian_loglik(Y_true, pred, Y_mask, sigma=float(args.std))  # [K,B]
            analytic_kl = kl_q_standard_normal(qmu, qlv)                           # [B]
            loss = iw_elbo_loss(logpx, analytic_kl, K)
            loss.backward(); optimizer.step()

        # validate
        rec.eval(); dec.eval()
        val_res = evaluate_mse(rec, dec, data_obj["val_dataloader"], data_obj["n_val_batches"], INPUT_DIM, DEVICE, num_sample=1)
        if val_res["mse"] < best_val:
            best_val = val_res["mse"]; best_iter = it
            test_res = evaluate_mse(rec, dec, data_obj["test_dataloader"], data_obj["n_test_batches"], INPUT_DIM, DEVICE, num_sample=1)
            test_report = {"mse": test_res["mse"], "rmse": test_res["rmse"]}
            best_path = REPO_ROOT / f"saved_models/mTAN_{ds_name}_{int(SystemRandom().random()*1e7)}.best.pt"
            latest_path = REPO_ROOT / f"saved_models/mTAN_{ds_name}.latest.pt"
            (REPO_ROOT / "saved_models").mkdir(parents=True, exist_ok=True)
            torch.save({"rec_state_dict": rec.state_dict(),
                        "dec_state_dict": dec.state_dict(),
                        "args": vars(args), "input_dim": INPUT_DIM}, best_path)
        scheduler.step(val_res["loss"])

        dt = time.time() - t0
        print(
            f"- Epoch {it:03d} | train_loss(one-batch): {loss.item():.6f} | "
            f"val_mse: {val_res['mse']:.6f} | val_rmse: {val_res['rmse']:.6f} | "
            + (f"best@{best_iter} test_mse: {test_report['mse']:.6f} rmse: {test_report['rmse']:.6f} | " if test_report else "")
            + f"time: {dt:.2f}s"
        )
        if writer:
            writer.add_scalar("train/loss_one_batch", float(loss.item()), it)
            writer.add_scalar("val/mse",  float(val_res["mse"]),  it)
            writer.add_scalar("val/rmse", float(val_res["rmse"]), it)
            if test_report:
                writer.add_scalar("test/mse_best",  float(test_report["mse"]),  it)
                writer.add_scalar("test/rmse_best", float(test_report["rmse"]), it)

except KeyboardInterrupt:
    print("\n[interrupt] stopping early")
    raise
params = {
    "niters": args.niters,
    "early_stop": args.early_stop,
    "batch_size": args.batch_size,
    "lr": args.lr,
    "latent_dim": args.latent_dim,
    "rec_hidden": args.rec_hidden,
    "gen_hidden": args.gen_hidden,
    "embed_time": args.embed_time,
    "num_heads": args.num_heads,
    "k_iwae": args.k_iwae,
    "std": args.std,
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
    model_name="mTAN",
    dataset=ds_name,
    metrics=metrics,
    params=params,
)

if writer:
    writer.flush(); writer.close()
