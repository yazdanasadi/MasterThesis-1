#!/usr/bin/env python
# hpsearch_icfld.py  — runs IC-FLD sweeps: NO-CYCLE first, then CYCLE
import os, sys, time, itertools, subprocess, csv, argparse
from pathlib import Path

# --------------------- defaults (edit or override by CLI) ---------------------
DEFAULT_DATASETS       = ("ushcn")
DEFAULT_SEEDS          = (0,)
OBS_HOURS              = 24
EPOCHS                 = 70
EARLY_STOP             = 10
BATCH_SIZE             = 32
GPU                    = "0"
BASIS_FN               = "L"            # {C,L,Q,S}
TBON                   = True
LOGDIR                 = "runs"
RESUME                 = ""         # "" to disable resume
LEARN_RATE             = 1e-3
WEIGHT_DECAY           = 0.0
CYCLE_LENGTH           = 24
TIME_MAX_HOURS         = 48

# Search space:
HIDDEN_DIM             = (32, 128, 256, 512)  # -> -ld
ATTN_HEADS             = (4, 8)               # -> -nh
DEPTH                  = (2, 4)               # -> --depth
EMB_PER_HEAD           = (2, 4, 8)            # total -ed = heads * per_head
# -----------------------------------------------------------------------------


def find_trainer_dir() -> tuple[Path, Path]:
    """
    Return (trainer_dir, repo_root).
    Works whether this script is in repo root or in FLD_ICC/.
    """
    here = Path(__file__).resolve().parent
    candidates = [
        here,                     # next to trainer
        here / "FLD_ICC",         # script in repo root
        here.parent / "FLD_ICC",  # script inside FLD_ICC
    ]
    for d in candidates:
        if (d / "train_FLD_ICC.py").exists():
            repo_root = d.parent if d.name == "FLD_ICC" else d
            return d, repo_root
    raise FileNotFoundError("Could not find FLD_ICC/train_FLD_ICC.py. "
                            "Place this script in repo root or FLD_ICC/.")


def build_cmd(args, dset, seed, ld, heads, depth, emb_per_head, use_cycle):
    ed_total = heads * emb_per_head
    cmd = [
        sys.executable, "train_FLD_ICC.py",
        "-d", dset,
        "-ot", str(args.obs),
        "-bs", str(args.batch_size),
        "--epochs", str(args.epochs),
        "--early-stop", str(args.early_stop),
        "--gpu", args.gpu,
        "-fn", args.fn,
        "-ed", str(ed_total),
        "-ld", str(ld),
        "-nh", str(heads),
        "--depth", str(depth),
        "--lr", str(args.lr),
        "--wd", str(args.wd),
        "--seed", str(seed),
    ]
    if args.tbon:
        cmd += ["--tbon", "--logdir", args.logdir]
    if args.resume:
        cmd += ["--resume", args.resume]
    if use_cycle:
        cmd += ["--use-cycle", "--cycle-length", str(args.cycle_length),
                "--time-max-hours", str(args.time_max_hours)]
    return cmd


def write_plan_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[plan] {path} ({len(rows)} runs)")


def run_phase(phase_name, trainer_dir: Path, rows, args):
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    total = len(rows)
    for i, r in enumerate(rows, 1):
        cmd = build_cmd(
            args,
            dset=r["dataset"], seed=r["seed"],
            ld=r["latent_dim"], heads=r["heads"], depth=r["depth"],
            emb_per_head=r["emb_per_head"], use_cycle=r["use_cycle"],
        )
        print("-" * 88)
        print(f"[{phase_name} {i}/{total}] cd {trainer_dir.name} &&", " ".join(cmd))
        print("-" * 88)
        if args.dry_run:
            continue
        proc = subprocess.run(cmd, cwd=str(trainer_dir), env=env)
        if proc.returncode != 0:
            print(f"[WARN] run failed (code {proc.returncode}), continuing…")


def main():
    p = argparse.ArgumentParser(description="IC-FLD sweep: no-cycle first, then with-cycle.")
    p.add_argument("--datasets", default=",".join(DEFAULT_DATASETS),
                   help="comma-separated: physionet,mimic,ushcn")
    p.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)),
                   help="comma-separated seeds, e.g., 0,1,2")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--early-stop", type=int, default=EARLY_STOP)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--obs", type=int, default=OBS_HOURS)
    p.add_argument("--fn", default=BASIS_FN, choices=["C","L","Q","S"])
    p.add_argument("--lr", type=float, default=LEARN_RATE)
    p.add_argument("--wd", type=float, default=WEIGHT_DECAY)
    p.add_argument("--gpu", default=GPU)
    p.add_argument("--tbon", action="store_true", default=TBON)
    p.add_argument("--logdir", default=LOGDIR)
    p.add_argument("--resume", default=RESUME)
    p.add_argument("--cycle-length", type=int, default=CYCLE_LENGTH)
    p.add_argument("--time-max-hours", type=int, default=TIME_MAX_HOURS)
    p.add_argument("--dry-run", action="store_true", help="print only, do not execute")
    # override search space if needed
    p.add_argument("--latent-dims", default=",".join(map(str, HIDDEN_DIM)))
    p.add_argument("--heads", default=",".join(map(str, ATTN_HEADS)))
    p.add_argument("--depths", default=",".join(map(str, DEPTH)))
    p.add_argument("--emb-per-head", default=",".join(map(str, EMB_PER_HEAD)))
    args = p.parse_args()

    datasets = tuple(s.strip() for s in args.datasets.split(",") if s.strip())
    seeds    = tuple(int(s) for s in args.seeds.split(",") if s.strip())
    lds      = tuple(int(s) for s in args.latent_dims.split(",") if s.strip())
    heads    = tuple(int(s) for s in args.heads.split(",") if s.strip())
    depths   = tuple(int(s) for s in args.depths.split(",") if s.strip())
    ephs     = tuple(int(s) for s in args.emb_per_head.split(",") if s.strip())

    trainer_dir, repo_root = find_trainer_dir()

    # Build plans
    ts = time.strftime("%Y%m%d_%H%M%S")
    plan_dir = repo_root / "joblogs"
    rows_a = [
        {"phase": "nocycle", "dataset": d, "seed": s, "latent_dim": ld,
         "heads": h, "depth": dp, "emb_per_head": eph, "use_cycle": False,
         "embedding_dim_total": h*eph}
        for d, s, ld, h, dp, eph in itertools.product(datasets, seeds, lds, heads, depths, ephs)
    ]
    rows_b = [
        {"phase": "cycle", "dataset": d, "seed": s, "latent_dim": ld,
         "heads": h, "depth": dp, "emb_per_head": eph, "use_cycle": True,
         "embedding_dim_total": h*eph}
        for d, s, ld, h, dp, eph in itertools.product(datasets, seeds, lds, heads, depths, ephs)
    ]

    write_plan_csv(plan_dir / f"icfld_sweep_nocycle_{ts}.csv", rows_a)
    write_plan_csv(plan_dir / f"icfld_sweep_cycle_{ts}.csv", rows_b)

    # Run NO-CYCLE first, then CYCLE
    run_phase("NO-CYCLE", trainer_dir, rows_a, args)
    run_phase("CYCLE",    trainer_dir, rows_b, args)


if __name__ == "__main__":
    main()
