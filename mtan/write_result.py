# write_result.py
from __future__ import annotations
import os, time, json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import pandas as pd

# Where results will be collected
DEFAULT_RESULTS_DIR  = Path("results")
DEFAULT_RESULTS_XLSX = DEFAULT_RESULTS_DIR / "training_results.xlsx"
DEFAULT_RESULTS_CSV  = DEFAULT_RESULTS_DIR / "training_results.csv"

def _acquire_lock(lock_path: Path, timeout: float = 120.0, poll: float = 0.25) -> bool:
    """Simple cross-platform lock via exclusive file creation."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return True
        except FileExistsError:
            time.sleep(poll)
    return False

def _release_lock(lock_path: Path):
    try:
        lock_path.unlink(missing_ok=True)
    except Exception:
        pass

def _flatten(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten nested dicts/lists/tuples into a single-level dict with string values."""
    flat = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            flat.update(_flatten(v, prefix=f"{key}."))
        elif isinstance(v, (list, tuple)):
            flat[key] = json.dumps(v)
        else:
            flat[key] = v
    return flat

def write_result(
    model_name: str,
    dataset: str,
    metrics: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]]  = None,
    *,
    output_file: Path | str = DEFAULT_RESULTS_XLSX,
    run_id: Optional[str] = None,
    notes: Optional[str] = None,
) -> Path:
    """
    Append one row with metrics & params to a shared Excel file + CSV mirror.
    Creates results/ if missing. Adds new columns on-the-fly.
    """
    output_file = Path(output_file)
    results_dir = output_file.parent
    results_dir.mkdir(parents=True, exist_ok=True)
    lock_path = results_dir / (output_file.stem + ".lock")

    # Prepare row
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    base = {
        "timestamp": now,
        "model": model_name,
        "dataset": dataset,
        "run_id": run_id,
    }
    row = {**base, **_flatten(params or {}), **_flatten(metrics or {})}
    if notes:
        row["notes"] = notes

    # Read existing Excel (if present)
    existing = None
    if output_file.exists():
        try:
            existing = pd.read_excel(output_file, engine="openpyxl")
        except Exception:
            existing = None

    # Build new dataframe with column union
    new_df = pd.DataFrame([row])
    if existing is not None and not existing.empty:
        # Union columns, keep stable order: base -> existing -> new extras
        base_cols = list(base.keys())
        existing_cols = [c for c in existing.columns if c not in base_cols]
        new_cols = [c for c in new_df.columns if c not in base_cols + existing_cols]
        all_cols = base_cols + existing_cols + new_cols
        existing = existing.reindex(columns=all_cols)
        new_df   = new_df.reindex(columns=all_cols)
        out_df = pd.concat([existing, new_df], ignore_index=True)
    else:
        out_df = new_df

    # Try to lock & write Excel; always mirror to CSV as well
    acquired = _acquire_lock(lock_path)
    try:
        # Excel
        try:
            with pd.ExcelWriter(output_file, engine="openpyxl", mode="w") as xw:
                out_df.to_excel(xw, index=False, sheet_name="Results")
        except Exception as e:
            print(f"[write_result] Excel write failed: {e}. Will still update CSV.")

        # CSV mirror
        out_df.to_csv(DEFAULT_RESULTS_CSV, index=False)

    finally:
        if acquired:
            _release_lock(lock_path)

    return output_file
