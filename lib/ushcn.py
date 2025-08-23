# lib/ushcn.py
import os
import glob
import numpy as np  # kept for compatibility
import pandas as pd
import torch

import lib.utils as utils
from torch.utils.data import Dataset  # kept for compatibility
from torch.nn.utils.rnn import pad_sequence
from typing import Optional
from pathlib import Path

class USHCN(object):
    """
    variables:
        "SNOW","SNWD","PRCP","TMAX","TMIN"
    """

    def __init__(
        self,
        root: str,
        n_samples: Optional[int] = None,
        device: torch.device = torch.device("cpu"),
        force_process: bool = False,
        raw_filename: str = "small_chunked_sporadic.csv",
    ):
        self.root = str(Path(root).resolve())
        self.device = device
        self.raw_filename = raw_filename

        # Prefer processed .pt; only process if missing or forced
        proc_path = self._find_processed()
        if force_process or (proc_path is None):
            self.process()  # will raise with a clear message if raw is missing
            proc_path = os.path.join(self.processed_folder, "ushcn.pt")

        # Load processed file
        if device == torch.device("cpu"):
            self.data = torch.load(proc_path, map_location="cpu")
        else:
            self.data = torch.load(proc_path)

        if n_samples is not None:
            print("Total records:", len(self.data))
            self.data = self.data[:n_samples]

    # ------------------------------- processing -------------------------------

    def process(self) -> None:
        """Create processed .pt from the raw CSV, if not already present."""
        if self._check_exists():
            return  # already have processed data

        os.makedirs(self.processed_folder, exist_ok=True)
        filename = os.path.join(self.raw_folder, self.raw_filename)
        print(f"Processing {filename}...")

        # Friendly error if raw is missing
        if not os.path.exists(filename):
            raise FileNotFoundError(
                f"[USHCN] Raw CSV not found at {filename}. "
                f"If you already have processed data, ensure "
                f"'{self.processed_folder}/ushcn.pt' (or any '*.pt') exists."
            )

        full_data = pd.read_csv(filename, index_col=0)
        full_data.index = full_data.index.astype("int32")

        entities = []
        value_cols = [c for c in full_data.columns if c.startswith("Value")]
        mask_cols = [("Mask" + x[5:]) for x in value_cols]

        # group by record id (index level 0)
        for record_id, data in full_data.groupby(level=0):
            # scale times to ~48h range like original codebase
            tt = torch.tensor(data["Time"].values, dtype=torch.float32, device=self.device) * (48.0 / 200.0)
            sorted_inds = tt.argsort()

            vals = torch.tensor(data[value_cols].values, dtype=torch.float32, device=self.device)
            mask = torch.tensor(data[mask_cols].values, dtype=torch.float32, device=self.device)

            # NaN-safe: zero out NaNs in vals and set mask=0 there
            if torch.isnan(vals).any():
                nan_mask = torch.isnan(vals)
                vals = vals.clone()
                mask = mask.clone()
                vals[nan_mask] = 0.0
                mask[nan_mask] = 0.0

            entities.append((record_id, tt[sorted_inds], vals[sorted_inds], mask[sorted_inds]))

        torch.save(entities, os.path.join(self.processed_folder, "ushcn.pt"))
        print("Total records:", len(entities))
        print("Done!")

    # ------------------------------- helpers ----------------------------------

    def _find_processed(self) -> Optional[str]:
        """Return a processed .pt path if it exists, else None."""
        preferred = os.path.join(self.processed_folder, "ushcn.pt")
        if os.path.exists(preferred):
            return preferred
        # fallback: any .pt in processed/
        cands = sorted(glob.glob(os.path.join(self.processed_folder, "*.pt")))
        return cands[0] if len(cands) > 0 else None

    def _check_exists(self) -> bool:
        """True if any processed .pt exists."""
        return self._find_processed() is not None

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, "raw")

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, "processed")

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


# --------------------------- batching utilities ---------------------------

def USHCN_time_chunk(data, args, device):
    chunk_data = []
    for _, (record_id, tt, vals, mask) in enumerate(data):
        for st in range(0, args.n_months - args.history - args.pred_window + 1, args.pred_window):
            et = st + args.history + args.pred_window
            if et == args.n_months:
                indices = torch.where((tt >= st) & (tt <= et))[0]
            else:
                indices = torch.where((tt >= st) & (tt < et))[0]
            t_bias = torch.tensor(st, device=device)
            chunk_data.append((record_id, tt[indices] - t_bias, vals[indices], mask[indices], t_bias))
    return chunk_data


def USHCN_patch_variable_time_collate_fn(
    batch,
    args,
    device=torch.device("cpu"),
    data_type="train",
    data_min=None,
    data_max=None,
    time_max=None,
):
    """
    Build patched inputs/targets for variable-time batches.
    """
    D = batch[0][2].shape[1]
    combined_tt, inverse_indices = torch.unique(
        torch.cat([ex[1] for ex in batch]), sorted=True, return_inverse=True
    )

    # number of observed time points
    n_observed_tp = torch.lt(combined_tt, args.history).sum()
    observed_tp = combined_tt[:n_observed_tp]

    # patch windows over observed_tp
    patch_indices = []
    st, ed = 0, args.patch_size
    for i in range(args.npatch):
        if i == args.npatch - 1:
            inds = torch.where((observed_tp >= st) & (observed_tp <= ed))[0]
        else:
            inds = torch.where((observed_tp >= st) & (observed_tp < ed))[0]
        patch_indices.append(inds)
        st += args.stride
        ed += args.stride

    offset = 0
    combined_vals = torch.zeros([len(batch), len(combined_tt), D], device=device)
    combined_mask = torch.zeros([len(batch), len(combined_tt), D], device=device)
    predicted_tp, predicted_data, predicted_mask, batch_t_bias = [], [], [], []

    for b, (record_id, tt, vals, mask, t_bias) in enumerate(batch):
        batch_t_bias.append(t_bias)
        indices = inverse_indices[offset : offset + len(tt)]
        offset += len(tt)

        combined_vals[b, indices] = vals
        combined_mask[b, indices] = mask

        tmp_n_observed_tp = torch.lt(tt, args.history).sum()
        predicted_tp.append(tt[tmp_n_observed_tp:])
        predicted_data.append(vals[tmp_n_observed_tp:])
        predicted_mask.append(mask[tmp_n_observed_tp:])

    combined_tt = combined_tt[:n_observed_tp]
    combined_vals = combined_vals[:, :n_observed_tp]
    combined_mask = combined_mask[:, :n_observed_tp]
    predicted_tp = pad_sequence(predicted_tp, batch_first=True)
    predicted_data = pad_sequence(predicted_data, batch_first=True)
    predicted_mask = pad_sequence(predicted_mask, batch_first=True)

    combined_tt = utils.normalize_masked_tp(combined_tt, att_min=0, att_max=time_max)
    predicted_tp = utils.normalize_masked_tp(predicted_tp, att_min=0, att_max=time_max)

    batch_t_bias = torch.stack(batch_t_bias)  # (B,)
    batch_t_bias = utils.normalize_masked_tp(batch_t_bias, att_min=0, att_max=time_max)

    data_dict = {
        "data": combined_vals,                # (B, T_o, D)
        "time_steps": combined_tt,            # (T_o,)
        "mask": combined_mask,                # (B, T_o, D)
        "data_to_predict": predicted_data,    # (B, T, D)
        "tp_to_predict": predicted_tp,        # (B, T)
        "mask_predicted_data": predicted_mask # (B, T, D)
    }

    data_dict = utils.split_and_patch_batch(data_dict, args, n_observed_tp, patch_indices)

    data_dict["observed_tp"] = data_dict["observed_tp"] + batch_t_bias.view(len(batch_t_bias), 1, 1, 1)
    data_dict["tp_to_predict"] = data_dict["tp_to_predict"] + batch_t_bias.view(len(batch_t_bias), 1)
    data_dict["tp_to_predict"][data_dict["mask_predicted_data"].sum(dim=-1) < 1e-8] = 0

    return data_dict


def USHCN_variable_time_collate_fn(
    batch,
    args,
    device=torch.device("cpu"),
    data_type="train",
    data_min=None,
    data_max=None,
    time_max=None,
):
    """
    Build non-patched (flat) variable-time batches.
    """
    observed_tp, observed_data, observed_mask = [], [], []
    predicted_tp, predicted_data, predicted_mask = [], [], []

    for _, (record_id, tt, vals, mask, t_bias) in enumerate(batch):
        n_observed_tp = torch.lt(tt, args.history).sum()
        tt = tt + t_bias

        observed_tp.append(tt[:n_observed_tp])
        observed_data.append(vals[:n_observed_tp])
        observed_mask.append(mask[:n_observed_tp])

        predicted_tp.append(tt[n_observed_tp:])
        predicted_data.append(vals[n_observed_tp:])
        predicted_mask.append(mask[n_observed_tp:])

    observed_tp = pad_sequence(observed_tp, batch_first=True)
    observed_data = pad_sequence(observed_data, batch_first=True)
    observed_mask = pad_sequence(observed_mask, batch_first=True)
    predicted_tp = pad_sequence(predicted_tp, batch_first=True)
    predicted_data = pad_sequence(predicted_data, batch_first=True)
    predicted_mask = pad_sequence(predicted_mask, batch_first=True)

    observed_tp = utils.normalize_masked_tp(observed_tp, att_min=0, att_max=time_max)
    predicted_tp = utils.normalize_masked_tp(predicted_tp, att_min=0, att_max=time_max)

    return {
        "observed_data": observed_data,
        "observed_tp": observed_tp,
        "observed_mask": observed_mask,
        "data_to_predict": predicted_data,
        "tp_to_predict": predicted_tp,
        "mask_predicted_data": predicted_mask,
    }


def USHCN_get_seq_length(args, records):
    max_input_len = 0
    max_pred_len = 0
    lens = []
    for _, (record_id, tt, vals, mask, t_bias) in enumerate(records):
        n_observed_tp = torch.lt(tt, args.history).sum()
        max_input_len = max(max_input_len, n_observed_tp)
        max_pred_len = max(max_pred_len, len(tt) - n_observed_tp)
        lens.append(n_observed_tp)
    lens = torch.stack(lens, dim=0)
    median_len = lens.median()
    return max_input_len, max_pred_len, median_len
