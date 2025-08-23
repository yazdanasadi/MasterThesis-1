#!/usr/bin/env python
# coding: utf-8

import os, sys, time, datetime, argparse, numpy as np, pandas as pd, random
from random import SystemRandom
from sklearn import model_selection
from write_result import write_result

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

sys.path.append("..")
import lib.utils as utils
from lib.parse_datasets import parse_datasets
from model.tPatchGNN import *
from lib.evaluation import evaluation, compute_all_losses

parser = argparse.ArgumentParser('IMTS Forecasting')
parser.add_argument('--state', type=str, default='def')
parser.add_argument('-n',  type=int, default=int(1e8), help="Size of the dataset")
parser.add_argument('--hop', type=int, default=1)
parser.add_argument('--nhead', type=int, default=1)
parser.add_argument('--tf_layer', type=int, default=1)
parser.add_argument('--nlayer', type=int, default=1)
parser.add_argument('--epoch', type=int, default=1000)
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--history', type=int, default=24)
parser.add_argument('-ps', '--patch_size', type=float, default=24)
parser.add_argument('--stride', type=float, default=24)
parser.add_argument('--logmode', type=str, default="a")

parser.add_argument('--lr',  type=float, default=1e-3)
parser.add_argument('--w_decay', type=float, default=0.0)
parser.add_argument('-b', '--batch_size', type=int, default=32)

parser.add_argument('--save', type=str, default='experiments/')
parser.add_argument('--load', type=str, default=None)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--dataset', type=str, default='physionet')
parser.add_argument('--quantization', type=float, default=0.0)
parser.add_argument('--model', type=str, default='tPatchGNN')
parser.add_argument('--outlayer', type=str, default='Linear')
parser.add_argument('-hd', '--hid_dim', type=int, default=64)
parser.add_argument('-td', '--te_dim', type=int, default=10)
parser.add_argument('-nd', '--node_dim', type=int, default=10)
parser.add_argument('--gpu', type=str, default='0')
# TensorBoard
parser.add_argument('--tbon', action='store_true', help='Enable TensorBoard logging')
parser.add_argument('--logdir', type=str, default='runs', help='TensorBoard log root')

args = parser.parse_args()
args.npatch = int(np.ceil((args.history - args.patch_size) / args.stride)) + 1

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
file_name = os.path.basename(__file__)[:-3]
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.PID = os.getpid()
print("PID, device:", args.PID, args.device)

if __name__ == '__main__':
    utils.setup_seed(args.seed)

    experimentID = args.load if args.load is not None else int(SystemRandom().random()*100000)
    ckpt_path = os.path.join(args.save, "experiment_" + str(experimentID) + '.ckpt')

    input_command = sys.argv
    ind = [i for i in range(len(input_command)) if input_command[i] == "--load"]
    if len(ind) == 1:
        ind = ind[0]
        input_command = input_command[:ind] + input_command[(ind+2):]
    input_command = " ".join(input_command)

    data_obj = parse_datasets(args, patch_ts=True)
    input_dim = data_obj["input_dim"]
    args.ndim = input_dim
    model = tPatchGNN(args).to(args.device)

    if(args.n < 12000):
        args.state = "debug"
        log_path = "logs/{}_{}_{}.log".format(args.dataset, args.model, args.state)
    else:
        log_path = "logs/{}_{}_{}_{}patch_{}stride_{}layer_{}lr.log".format(
            args.dataset, args.model, args.state, args.patch_size, args.stride, args.nlayer, args.lr
        )
    if not os.path.exists("logs/"): utils.makedirs("logs/")
    logger = utils.get_logger(logpath=log_path, filepath=os.path.abspath(__file__), mode=args.logmode)
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info(input_command)
    logger.info(args)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    num_batches = data_obj["n_train_batches"]
    print("n_train_batches:", num_batches)

    # ---- TensorBoard (optional) ----
    writer = None
    if args.tbon:
        run_name = f"tPatchGNN_{args.dataset}_{int(time.time())}"
        writer = SummaryWriter(log_dir=os.path.join(args.logdir, run_name))

    best_val_mse = np.inf
    test_res = None
    for itr in range(args.epoch):
        st = time.time()

        model.train()
        for _ in range(num_batches):
            optimizer.zero_grad()
            batch_dict = utils.get_next_batch(data_obj["train_dataloader"])
            train_res = compute_all_losses(model, batch_dict)
            train_res["loss"].backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_res = evaluation(model, data_obj["val_dataloader"], data_obj["n_val_batches"])
            if(val_res["mse"] < best_val_mse):
                best_val_mse = val_res["mse"]
                best_iter = itr
                test_res = evaluation(model, data_obj["test_dataloader"], data_obj["n_test_batches"])

            logger.info('- Epoch {:03d}, ExpID {}'.format(itr, experimentID))
            logger.info("Train - Loss (one batch): {:.5f}".format(train_res["loss"].item()))
            logger.info("Val - Loss, MSE, RMSE, MAE, MAPE: {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.2f}%".format(
                val_res["loss"], val_res["mse"], val_res["rmse"], val_res["mae"], val_res["mape"]*100))
            if(test_res != None):
                logger.info("Test - Best epoch, Loss, MSE, RMSE, MAE, MAPE: {}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.2f}%".format(
                    best_iter, test_res["loss"], test_res["mse"], test_res["rmse"], test_res["mae"], test_res["mape"]*100))
            logger.info("Time spent: {:.2f}s".format(time.time()-st))

            if writer:
                writer.add_scalar("train/loss_one_batch", float(train_res["loss"].item()), itr)
                writer.add_scalar("val/loss",  float(val_res["loss"]),  itr)
                writer.add_scalar("val/mse",   float(val_res["mse"]),   itr)
                writer.add_scalar("val/rmse",  float(val_res["rmse"]),  itr)
                writer.add_scalar("val/mae",   float(val_res["mae"]),   itr)
                writer.add_scalar("val/mape",  float(val_res["mape"]),  itr)
                if test_res is not None:
                    writer.add_scalar("test/loss_best", float(test_res["loss"]), itr)
                    writer.add_scalar("test/mse_best",  float(test_res["mse"]),  itr)
                    writer.add_scalar("test/rmse_best", float(test_res["rmse"]), itr)
                    writer.add_scalar("test/mae_best",  float(test_res["mae"]),  itr)
                    writer.add_scalar("test/mape_best", float(test_res["mape"]), itr)

        if(itr - best_iter >= args.patience):
            print("Exp has been early stopped!")
            break
    params = {
        "epoch": args.epoch,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "w_decay": args.w_decay,
        "nhead": args.nhead,
        "tf_layer": args.tf_layer,
        "nlayer": args.nlayer,
        "patch_size": args.patch_size,
        "stride": args.stride,
        "hid_dim": args.hid_dim,
        "te_dim": args.te_dim,
        "node_dim": args.node_dim,
        "seed": args.seed,
        "history": args.history,
    }
    metrics = {
        "best_epoch": (best_iter if 'best_iter' in locals() else None),
        "val_mse_best": float(best_val_mse),
        "val_rmse_best": float((best_val_mse + 1e-8) ** 0.5),
        "train_loss_last_batch": float(train_res["loss"].item()) if 'train_res' in locals() else None,
        "test_mse_best": (float(test_res["mse"]) if test_res else None),
        "test_rmse_best": (float(test_res["rmse"]) if test_res else None),
        "test_mae_best": (float(test_res["mae"]) if test_res else None),
        "test_mape_best": (float(test_res["mape"]) if test_res else None),
    }

    # ---- ALWAYS write the shared Excel/CSV result ----
    write_result(
        model_name="t-PatchGNN",
        dataset=args.dataset,
        metrics=metrics,
        params=params,
        run_id=str(experimentID),
    )


    if writer:

        writer.flush(); writer.close()
