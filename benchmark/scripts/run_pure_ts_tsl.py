#!/usr/bin/env python3
"""Run pure time-series baselines for the Hebei-only PV-Text benchmark.

This runner is intentionally framework-conservative: all listed models are
launched through Time-Series-Library with the same Dataset_Custom loader,
StandardScaler preprocessing, train/validation/test split, target column, and
metric extraction.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd


STATIONS = [f"hebei_station{i:02d}" for i in range(10)]
DEFAULT_MODELS = [
    "iTransformer",
    "PatchTST",
    "TimeXer",
    "TimesNet",
    "DLinear",
    "TimeMixer",
    "Informer",
]
DEFAULT_PRED_LENS = [16, 32, 48, 96]


def parse_csv_list(value: str, cast=str):
    return [cast(x.strip()) for x in value.split(",") if x.strip()]


def visible_gpus() -> list[int]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        gpus = [int(x.strip()) for x in out.splitlines() if x.strip().isdigit()]
        return gpus or [0]
    except Exception:
        return [0]


def load_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def result_dir_for(tsl_root: Path, model_id: str) -> Path:
    matches = sorted(
        (tsl_root / "results").glob(f"*{model_id}*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        raise FileNotFoundError(model_id)
    return matches[0]


def target_metrics(result_dir: Path) -> tuple[float, float, float, tuple[int, ...], tuple[int, ...]]:
    pred = np.load(result_dir / "pred.npy")
    true = np.load(result_dir / "true.npy")
    pred_t = pred[..., -1]
    true_t = true[..., -1]
    err = pred_t - true_t
    mse = float(np.mean(err**2))
    mae = float(np.mean(np.abs(err)))
    return mae, mse, float(np.sqrt(mse)), tuple(pred.shape), tuple(true.shape)


def command_for(args, model: str, station: str, pred_len: int) -> tuple[str, list[str]]:
    model_id = f"pvtext_ts_{model}_{station}_pl{pred_len}"
    cmd = [
        args.python,
        "-u",
        "run.py",
        "--task_name",
        "long_term_forecast",
        "--is_training",
        "1",
        "--model_id",
        model_id,
        "--model",
        model,
        "--data",
        "custom",
        "--root_path",
        str(args.tsl_data_dir) + "/",
        "--data_path",
        f"{station}.csv",
        "--features",
        "M",
        "--target",
        "OT",
        "--freq",
        "t",
        "--seq_len",
        str(args.seq_len),
        "--label_len",
        str(args.label_len),
        "--pred_len",
        str(pred_len),
        "--enc_in",
        "15",
        "--dec_in",
        "15",
        "--c_out",
        "15",
        "--d_model",
        str(args.d_model),
        "--n_heads",
        str(args.n_heads),
        "--e_layers",
        str(args.e_layers),
        "--d_layers",
        str(args.d_layers),
        "--d_ff",
        str(args.d_ff),
        "--dropout",
        str(args.dropout),
        "--batch_size",
        str(args.batch_size),
        "--train_epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--learning_rate",
        str(args.learning_rate),
        "--num_workers",
        str(args.num_workers),
        "--des",
        "pvtext_hebei_ts",
        "--itr",
        "1",
        "--gpu",
        "0",
    ]
    if model == "TimeMixer":
        cmd += [
            "--channel_independence",
            "0",
            "--down_sampling_method",
            "avg",
            "--down_sampling_layers",
            "1",
            "--down_sampling_window",
            "2",
        ]
    if model == "TimeXer":
        cmd += ["--patch_len", "16"]
    return model_id, cmd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default="/anaconda3/envs/itransformer/bin/python3.8")
    parser.add_argument("--tsl_root", type=Path, default=Path("/root/Time-Series-Library"))
    parser.add_argument("--data_root", type=Path, default=Path("/root/KDD/hebei_daytime_0800_1900"))
    parser.add_argument("--output_dir", type=Path, default=Path("/root/KDD/experiments_pvtext_pure_ts"))
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--stations", default=",".join(STATIONS))
    parser.add_argument("--pred_lens", default=",".join(map(str, DEFAULT_PRED_LENS)))
    parser.add_argument("--gpus", default="")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--label_len", type=int, default=48)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--e_layers", type=int, default=2)
    parser.add_argument("--d_layers", type=int, default=1)
    parser.add_argument("--d_ff", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.tsl_data_dir = args.tsl_root / "dataset" / "pvtext_hebei_ts"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "logs").mkdir(parents=True, exist_ok=True)
    args.tsl_data_dir.mkdir(parents=True, exist_ok=True)

    models = parse_csv_list(args.models)
    stations = parse_csv_list(args.stations)
    pred_lens = parse_csv_list(args.pred_lens, int)
    gpus = parse_csv_list(args.gpus, int) if args.gpus else visible_gpus()
    lock = threading.Lock()
    summary_path = args.output_dir / "summary_pure_ts.csv"

    for station in stations:
        shutil.copy2(args.data_root / station / "solar.csv", args.tsl_data_dir / f"{station}.csv")

    manifest = vars(args).copy()
    manifest.update({"models": models, "stations": stations, "pred_lens": pred_lens, "gpus": gpus})
    manifest = {k: str(v) if isinstance(v, Path) else v for k, v in manifest.items()}
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def write_aggregate() -> None:
        df = load_summary(summary_path)
        if df.empty:
            return
        ok = df[df["returncode"] == 0].copy()
        if ok.empty:
            return
        ok.groupby(["model", "pred_len"]).agg(
            n=("station", "count"),
            mean_mae=("target_mae", "mean"),
            mean_mse=("target_mse", "mean"),
            mean_rmse=("target_rmse", "mean"),
            median_mae=("target_mae", "median"),
            median_mse=("target_mse", "median"),
        ).reset_index().to_csv(args.output_dir / "aggregate_by_horizon.csv", index=False)
        ok.groupby("model").agg(
            n=("station", "count"),
            mean_mae=("target_mae", "mean"),
            mean_mse=("target_mse", "mean"),
            mean_rmse=("target_rmse", "mean"),
            total_elapsed_sec=("elapsed_sec", "sum"),
        ).reset_index().to_csv(args.output_dir / "aggregate_overall.csv", index=False)

    def upsert(row: dict) -> None:
        with lock:
            old = load_summary(summary_path)
            if old.empty:
                new = pd.DataFrame([row])
            else:
                for col in row:
                    if col not in old.columns:
                        old[col] = np.nan
                mask = (
                    (old["model"] == row["model"])
                    & (old["station"] == row["station"])
                    & (old["pred_len"] == row["pred_len"])
                )
                if mask.any():
                    for col, value in row.items():
                        old.loc[mask, col] = value
                    new = old
                else:
                    new = pd.concat([old, pd.DataFrame([row])], ignore_index=True)
            new.sort_values(["model", "pred_len", "station"]).to_csv(summary_path, index=False)
            write_aggregate()

    def run_task(task):
        model, station, pred_len, gpu = task
        old = load_summary(summary_path)
        if not args.force and not old.empty:
            done = old[
                (old["model"] == model)
                & (old["station"] == station)
                & (old["pred_len"] == pred_len)
                & (old["returncode"] == 0)
            ]
            if len(done):
                return done.iloc[-1].to_dict()
        model_id, cmd = command_for(args, model, station, pred_len)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        started = time.time()
        proc = subprocess.run(
            cmd,
            cwd=str(args.tsl_root),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=21600,
        )
        elapsed = round(time.time() - started, 2)
        log_path = args.output_dir / "logs" / f"{model_id}.log"
        log_path.write_text(proc.stdout, encoding="utf-8", errors="replace")
        row = {
            "model": model,
            "station": station,
            "pred_len": pred_len,
            "model_id": model_id,
            "returncode": proc.returncode,
            "gpu": gpu,
            "elapsed_sec": elapsed,
            "log": str(log_path),
        }
        if proc.returncode == 0:
            try:
                result_dir = result_dir_for(args.tsl_root, model_id)
                mae, mse, rmse, pred_shape, true_shape = target_metrics(result_dir)
                row.update(
                    {
                        "target_mae": mae,
                        "target_mse": mse,
                        "target_rmse": rmse,
                        "result_dir": str(result_dir),
                        "pred_shape": str(pred_shape),
                        "true_shape": str(true_shape),
                    }
                )
            except Exception as exc:
                row.update({"returncode": 98, "metric_error": repr(exc)})
        else:
            row["error_tail"] = proc.stdout[-3000:].encode("ascii", "ignore").decode("ascii", "ignore")
        upsert(row)
        print(
            f"DONE model={model} station={station} pred_len={pred_len} "
            f"rc={row['returncode']} mae={row.get('target_mae')} mse={row.get('target_mse')} "
            f"gpu={gpu} elapsed={elapsed}s",
            flush=True,
        )
        return row

    tasks = []
    idx = 0
    for model in models:
        for pred_len in pred_lens:
            for station in stations:
                tasks.append((model, station, pred_len, gpus[idx % len(gpus)]))
                idx += 1
    print(f"START tasks={len(tasks)} gpus={gpus} output={args.output_dir}", flush=True)
    with futures.ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        for future in futures.as_completed([executor.submit(run_task, task) for task in tasks]):
            future.result()
    write_aggregate()
    print("ALL_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
