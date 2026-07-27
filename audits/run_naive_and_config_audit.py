import csv
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path("/root/KDD")
DATA = ROOT / "hebei_daytime_0800_1900"
OUT = ROOT / "audits_pv_text_benchmark"
OUT.mkdir(exist_ok=True)
STATIONS = [f"hebei_station{i:02d}" for i in range(10)]
PRED_LENS = [16, 32, 48, 96]
SEQ_LEN = 96


def borders(n):
    n_train = int(n * 0.7)
    n_test = int(n * 0.2)
    n_val = n - n_train - n_test
    return [0, n_train - SEQ_LEN, n - n_test - SEQ_LEN], [n_train, n_train + n_val, n]


def metrics(pred, true):
    err = pred - true
    mse = float(np.mean(err ** 2))
    mae = float(np.mean(np.abs(err)))
    return mae, mse, float(np.sqrt(mse))


rows = []
for station in STATIONS:
    raw = pd.read_csv(DATA / station / "solar.csv")
    vals = raw.drop(columns=["date"]).astype("float32")
    if "OT" not in vals.columns:
        continue
    target_idx = list(vals.columns).index("OT")
    arr = vals.values
    n = len(arr)
    b1s, b2s = borders(n)
    scaler = StandardScaler()
    scaler.fit(arr[b1s[0]:b2s[0]])
    data = scaler.transform(arr)
    target = data[:, target_idx]
    b1, b2 = b1s[2], b2s[2]
    for pred_len in PRED_LENS:
        preds_last, preds_seasonal, trues = [], [], []
        max_i = b2 - b1 - SEQ_LEN - pred_len + 1
        for idx in range(max(0, max_i)):
            s = b1 + idx
            e = s + SEQ_LEN
            y0 = e
            y1 = y0 + pred_len
            true = target[y0:y1]
            preds_last.append(np.repeat(target[e - 1], pred_len))
            # 96 sub-hourly steps roughly mirrors the input-window daily-ish lag used in this benchmark setting.
            if s >= pred_len:
                seasonal = target[e - pred_len:e]
            else:
                seasonal = np.repeat(target[e - 1], pred_len)
            preds_seasonal.append(seasonal)
            trues.append(true)
        trues = np.stack(trues)
        for name, pred_list in [("last_value", preds_last), ("recent_window_copy", preds_seasonal)]:
            pred = np.stack(pred_list)
            mae, mse, rmse = metrics(pred, trues)
            rows.append({"model": name, "station": station, "pred_len": pred_len, "mae": mae, "mse": mse, "rmse": rmse, "n": len(trues)})

with (OUT / "naive_baseline_audit.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

print("NAIVE_BY_MODEL")
df = pd.DataFrame(rows)
print(df.groupby("model")[["mae", "mse", "rmse"]].mean().to_string())
print("NAIVE_BY_MODEL_PRED")
print(df.groupby(["model", "pred_len"])[["mae", "mse", "rmse"]].mean().to_string())

# Lightweight config/log audit.
ts = pd.read_csv(ROOT / "final_result_tables/summary_all_models_hebei_ts.csv")
audit_rows = []
for model, sub in ts.groupby("model"):
    sample = sub.iloc[0]
    log_path = Path(str(sample.get("log", "")))
    exists = log_path.exists()
    text = log_path.read_text(errors="ignore")[:5000] if exists else ""
    audit_rows.append({
        "model": model,
        "rows": len(sub),
        "pred_lens": sorted(sub["pred_len"].unique().tolist()),
        "stations": sub["station"].nunique(),
        "mean_mae": sub["target_mae"].mean(),
        "sample_log": str(log_path),
        "log_exists": exists,
        "has_args_or_setting": bool(re.search(r"args|Namespace|seq_len|pred_len|enc_in|features|target|scale", text, re.I)),
        "log_head": text[:500].replace("\n", " ") if text else "",
    })
with (OUT / "config_log_fairness_audit.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(audit_rows[0].keys()))
    w.writeheader()
    w.writerows(audit_rows)
print("CONFIG_AUDIT")
print(pd.DataFrame(audit_rows)[["model", "rows", "pred_lens", "stations", "mean_mae", "log_exists", "has_args_or_setting"]].to_string(index=False))
