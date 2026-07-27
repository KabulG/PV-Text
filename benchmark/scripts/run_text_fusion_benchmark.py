
import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
import traceback
from multiprocessing import Process, Queue
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, "/root/TimeCMA")
try:
    from models.TimeCMA import Dual as TimeCMADual
except Exception:
    TimeCMADual = None

ROOT = Path("/root/KDD/hebei_daytime_0800_1900")
OUT = Path("/root/KDD/experiments_text_fusion_models")
LOG_DIR = OUT / "logs"
RESULT_DIR = OUT / "results"
for p in [OUT, LOG_DIR, RESULT_DIR]:
    p.mkdir(parents=True, exist_ok=True)

STATIONS = [f"hebei_station{i:02d}" for i in range(10)]
PRED_LENS = [16, 32, 48, 64]
PHASES = ["text_stage1", "text_stage2", "text_stage12"]
MODELS = ["TimeCMA", "CALFAdapter", "TimeLLMAdapter", "TextFiLM"]


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.floor("h")
    return df


def build_text_map(station, phase):
    if phase == "text_stage1":
        df = read_jsonl(ROOT / station / "rt_text1.jsonl")
        cols = ["state_prompt", "recent_trend_prompt", "statistical_variability_prompt"]
        out = {}
        for _, r in df.iterrows():
            parts = [str(r[c]) for c in cols if c in r and pd.notna(r[c])]
            out[r["timestamp"]] = " ".join(parts)
        return out
    if phase == "text_stage2":
        df = read_jsonl(ROOT / station / "rt_text2.jsonl")
        cols = ["low_frequency_trend_prompt", "high_frequency_component_prompt"]
        out = {}
        for _, r in df.iterrows():
            parts = [str(r[c]) for c in cols if c in r and pd.notna(r[c])]
            out[r["timestamp"]] = " ".join(parts)
        return out
    if phase == "text_stage12":
        df1 = read_jsonl(ROOT / station / "rt_text1.jsonl")
        cols1 = ["state_prompt", "recent_trend_prompt", "statistical_variability_prompt"]
        df2 = read_jsonl(ROOT / station / "rt_text2.jsonl")
        cols2 = ["low_frequency_trend_prompt", "high_frequency_component_prompt"]
        out1 = {}
        for _, r in df1.iterrows():
            parts = [str(r[c]) for c in cols1 if c in r and pd.notna(r[c])]
            out1[r["timestamp"]] = " ".join(parts)
        out2 = {}
        for _, r in df2.iterrows():
            parts = [str(r[c]) for c in cols2 if c in r and pd.notna(r[c])]
            out2[r["timestamp"]] = " ".join(parts)
        out = {}
        for ts in sorted(set(out1) | set(out2)):
            out[ts] = " ".join([x for x in [out1.get(ts, ""), out2.get(ts, "")] if x])
        return out
    raise ValueError(phase)


def stable_text_embedding(text, dim=256):
    vec = np.zeros(dim, dtype=np.float32)
    toks = text.lower().replace("\n", " ").split()
    if not toks:
        toks = ["missing_text"]
    for tok in toks:
        h = hashlib.blake2b(tok.encode("utf-8", errors="ignore"), digest_size=8).digest()
        val = int.from_bytes(h, byteorder="little", signed=False)
        idx = val % dim
        sign = 1.0 if ((val >> 8) & 1) else -1.0
        vec[idx] += sign
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec /= norm
    return vec


def build_station_text_embeddings(station, phase, dates, dim=256):
    cache_dir = OUT / "text_embedding_cache" / phase
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{station}_hash{dim}.pt"
    hours = sorted(set(pd.Series(dates).dt.floor("h")))
    if cache_path.exists():
        pack = torch.load(cache_path, map_location="cpu")
        if pack.get("hours") == [str(h) for h in hours]:
            return {pd.Timestamp(h): pack["emb"][i] for i, h in enumerate(pack["hours"])}
    text_map = build_text_map(station, phase)
    vecs = []
    fallback = ""
    for h in hours:
        txt = text_map.get(h, fallback)
        if not txt:
            txt = "No textual condition is available for this hour."
        fallback = txt
        vecs.append(stable_text_embedding(txt, dim=dim))
    emb = torch.from_numpy(np.stack(vecs, axis=0))
    torch.save({"hours": [str(h) for h in hours], "emb": emb}, cache_path)
    return {pd.Timestamp(h): emb[i] for i, h in enumerate(hours)}


class PVTextDataset(Dataset):
    def __init__(self, station, phase, pred_len, flag, text_dim=256):
        self.station = station
        self.phase = phase
        self.pred_len = pred_len
        self.seq_len = 96
        raw = pd.read_csv(ROOT / station / "solar.csv")
        cols = list(raw.columns)
        cols.remove("date")
        cols.remove("OT")
        raw = raw[["date"] + cols + ["OT"]]
        raw["date"] = pd.to_datetime(raw["date"])
        self.dates = raw["date"].reset_index(drop=True)
        values = raw.drop(columns=["date"]).values.astype("float32")
        n = len(raw)
        n_train = int(n * 0.7)
        n_test = int(n * 0.2)
        n_val = n - n_train - n_test
        self.border1s = [0, n_train - self.seq_len, n - n_test - self.seq_len]
        self.border2s = [n_train, n_train + n_val, n]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]
        self.border1 = self.border1s[self.set_type]
        self.border2 = self.border2s[self.set_type]
        self.scaler = StandardScaler()
        self.scaler.fit(values[self.border1s[0]:self.border2s[0]])
        self.data = self.scaler.transform(values).astype("float32")
        self.num_nodes = self.data.shape[1]
        self.text_dim = text_dim
        self.hour_to_emb = build_station_text_embeddings(station, phase, self.dates, dim=text_dim)

    def __len__(self):
        return self.border2 - self.border1 - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        s = self.border1 + idx
        e = s + self.seq_len
        y0 = e
        y1 = y0 + self.pred_len
        x = self.data[s:e]
        y = self.data[y0:y1]
        mark = np.zeros((self.seq_len, 6), dtype="float32")
        last_hour = self.dates.iloc[e - 1].floor("h")
        base_emb = self.hour_to_emb[last_hour].float()
        node_emb = base_emb[:, None].repeat(1, self.num_nodes).unsqueeze(-1)
        return torch.from_numpy(x), torch.from_numpy(mark), node_emb, torch.from_numpy(y)


class TextFiLMNet(nn.Module):
    def __init__(self, num_nodes, pred_len, text_dim=256, hidden=128):
        super().__init__()
        self.pred_len = pred_len
        self.num_nodes = num_nodes
        self.encoder = nn.GRU(num_nodes, hidden, batch_first=True)
        self.film = nn.Sequential(nn.Linear(text_dim, hidden * 2), nn.GELU(), nn.Linear(hidden * 2, hidden * 2))
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, pred_len * num_nodes))

    def forward(self, x, mark, emb):
        _, h = self.encoder(x)
        h = h[-1]
        text = emb.squeeze(-1).mean(dim=2)
        gamma, beta = self.film(text).chunk(2, dim=-1)
        h = h * (1.0 + torch.tanh(gamma)) + beta
        return self.head(h).view(x.size(0), self.pred_len, self.num_nodes)


class CALFAdapterNet(nn.Module):
    def __init__(self, num_nodes, pred_len, text_dim=256, hidden=128, heads=4):
        super().__init__()
        self.pred_len = pred_len
        self.num_nodes = num_nodes
        self.ts_encoder = nn.GRU(num_nodes, hidden, batch_first=True)
        self.text_proj = nn.Linear(text_dim, hidden)
        self.cross_attn = nn.MultiheadAttention(hidden, heads, batch_first=True, dropout=0.1)
        self.gate = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, pred_len * num_nodes))

    def forward(self, x, mark, emb):
        _, h = self.ts_encoder(x)
        q = h[-1].unsqueeze(1)
        text_tokens = self.text_proj(emb.squeeze(-1).permute(0, 2, 1))
        ctx, _ = self.cross_attn(q, text_tokens, text_tokens, need_weights=False)
        ctx = ctx.squeeze(1)
        h0 = q.squeeze(1)
        g = self.gate(torch.cat([h0, ctx], dim=-1))
        fused = g * h0 + (1.0 - g) * ctx
        return self.head(fused).view(x.size(0), self.pred_len, self.num_nodes)


class TimeLLMAdapterNet(nn.Module):
    def __init__(self, num_nodes, pred_len, seq_len=96, text_dim=256, hidden=128, patch_len=16, stride=8, heads=4):
        super().__init__()
        self.pred_len = pred_len
        self.num_nodes = num_nodes
        self.patch_len = patch_len
        self.stride = stride
        self.patch_proj = nn.Linear(patch_len * num_nodes, hidden)
        self.text_proj = nn.Linear(text_dim, hidden)
        self.self_attn = nn.MultiheadAttention(hidden, heads, batch_first=True, dropout=0.1)
        self.cross_attn = nn.MultiheadAttention(hidden, heads, batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, pred_len * num_nodes)

    def forward(self, x, mark, emb):
        patches = []
        for start in range(0, x.size(1) - self.patch_len + 1, self.stride):
            patches.append(x[:, start:start + self.patch_len, :].reshape(x.size(0), -1))
        tok = self.patch_proj(torch.stack(patches, dim=1))
        tok2, _ = self.self_attn(tok, tok, tok, need_weights=False)
        tok = self.norm(tok + tok2)
        text = self.text_proj(emb.squeeze(-1).permute(0, 2, 1))
        ctx, _ = self.cross_attn(tok, text, text, need_weights=False)
        pooled = self.norm(tok + ctx).mean(dim=1)
        return self.head(pooled).view(x.size(0), self.pred_len, self.num_nodes)


class TimeCMANet(nn.Module):
    def __init__(self, num_nodes, pred_len, text_dim=256):
        super().__init__()
        if TimeCMADual is None:
            raise RuntimeError("TimeCMA import failed")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TimeCMADual(device=device, channel=64, num_nodes=num_nodes, seq_len=96,
                                 pred_len=pred_len, dropout_n=0.1, d_llm=text_dim,
                                 e_layer=1, d_layer=1, head=4)

    def forward(self, x, mark, emb):
        return self.model(x, mark, emb)


def build_model(model_name, num_nodes, pred_len, text_dim):
    if model_name == "TimeCMA":
        return TimeCMANet(num_nodes, pred_len, text_dim)
    if model_name == "CALFAdapter":
        return CALFAdapterNet(num_nodes, pred_len, text_dim)
    if model_name == "TimeLLMAdapter":
        return TimeLLMAdapterNet(num_nodes, pred_len, text_dim=text_dim)
    if model_name == "TextFiLM":
        return TextFiLMNet(num_nodes, pred_len, text_dim)
    raise ValueError(model_name)


def metric_target(pred, true):
    err = pred[:, :, -1] - true[:, :, -1]
    mse = torch.mean(err ** 2).item()
    mae = torch.mean(torch.abs(err)).item()
    return mae, mse, float(math.sqrt(max(mse, 0.0)))


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    preds, trues = [], []
    loss_fn = nn.MSELoss()
    with torch.no_grad():
        for x, mark, emb, y in loader:
            x = x.to(device)
            mark = mark.to(device)
            emb = emb.to(device)
            y = y.to(device)
            pred = model(x, mark, emb)
            total_loss += loss_fn(pred, y).item()
            n_batches += 1
            preds.append(pred.detach().cpu())
            trues.append(y.detach().cpu())
    pred = torch.cat(preds, dim=0)
    true = torch.cat(trues, dim=0)
    mae, mse, rmse = metric_target(pred, true)
    return total_loss / max(n_batches, 1), mae, mse, rmse


def run_one(task, gpu_id, args):
    station = task["station"]
    phase = task["phase"]
    pred_len = task["pred_len"]
    model_name = task["model"]
    result_path = RESULT_DIR / f"{model_name}_{phase}_{station}_pl{pred_len}.json"
    if result_path.exists() and not args.force:
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    seed_all(2027 + pred_len + int(station[-2:]))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    started = time.time()
    log_path = LOG_DIR / f"{model_name}_{phase}_{station}_pl{pred_len}.log"
    try:
        train_set = PVTextDataset(station, phase, pred_len, "train", args.text_dim)
        val_set = PVTextDataset(station, phase, pred_len, "val", args.text_dim)
        test_set = PVTextDataset(station, phase, pred_len, "test", args.text_dim)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=0)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=0)
        model = build_model(model_name, train_set.num_nodes, pred_len, args.text_dim).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        loss_fn = nn.MSELoss()
        best_val = float("inf")
        best_state = None
        bad = 0
        with log_path.open("w", encoding="utf-8") as log:
            log.write(json.dumps(task) + "\n")
            for epoch in range(1, args.epochs + 1):
                model.train()
                losses = []
                for x, mark, emb, y in train_loader:
                    x = x.to(device)
                    mark = mark.to(device)
                    emb = emb.to(device)
                    y = y.to(device)
                    opt.zero_grad()
                    pred = model(x, mark, emb)
                    loss = loss_fn(pred, y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    opt.step()
                    losses.append(loss.item())
                val_loss, val_mae, val_mse, val_rmse = evaluate(model, val_loader, device)
                train_loss = float(np.mean(losses)) if losses else float("nan")
                log.write(f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f} val_mae={val_mae:.6f}\n")
                log.flush()
                if val_loss < best_val:
                    best_val = val_loss
                    best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                    bad = 0
                else:
                    bad += 1
                if bad >= args.patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        test_loss, mae, mse, rmse = evaluate(model, test_loader, device)
        result = {
            **task,
            "status": "success",
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "val_loss": best_val,
            "test_loss": test_loss,
            "epochs": args.epochs,
            "elapsed_sec": round(time.time() - started, 2),
            "gpu": gpu_id,
            "text_dim": args.text_dim,
            "note": "Only one text phase is fused; embedding uses the last observed input hour to avoid future leakage.",
        }
    except Exception as exc:
        result = {
            **task,
            "status": "failed",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "elapsed_sec": round(time.time() - started, 2),
            "gpu": gpu_id,
        }
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


def worker(gpu_id, q, args):
    while True:
        task = q.get()
        if task is None:
            return
        run_one(task, gpu_id, args)


def write_summary():
    rows = []
    for path in sorted(RESULT_DIR.glob("*.json")):
        try:
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            pass
    fields = ["model", "phase", "station", "pred_len", "status", "mae", "mse", "rmse", "val_loss", "test_loss", "elapsed_sec", "gpu", "error"]
    with (OUT / "summary_text_fusion_models.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)
    ok = sum(1 for r in rows if r.get("status") == "success")
    fail = sum(1 for r in rows if r.get("status") == "failed")
    print(f"SUMMARY rows={len(rows)} success={ok} failed={fail} path={OUT / 'summary_text_fusion_models.csv'}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--phases", default=",".join(PHASES))
    ap.add_argument("--pred_lens", default=",".join(map(str, PRED_LENS)))
    ap.add_argument("--stations", default=",".join(STATIONS))
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--text_dim", type=int, default=256)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    models = [x for x in args.models.split(",") if x]
    phases = [x for x in args.phases.split(",") if x]
    pred_lens = [int(x) for x in args.pred_lens.split(",") if x]
    stations = [x for x in args.stations.split(",") if x]
    gpus = [int(x) for x in args.gpus.split(",") if x != ""]
    tasks = [{"model": m, "phase": ph, "station": st, "pred_len": pl}
             for m in models for ph in phases for st in stations for pl in pred_lens]
    if args.smoke:
        tasks = [
            {"model": "TimeCMA", "phase": "text_stage1", "station": "hebei_station00", "pred_len": 16},
            {"model": "CALFAdapter", "phase": "text_stage1", "station": "hebei_station00", "pred_len": 16},
            {"model": "TimeLLMAdapter", "phase": "text_stage2", "station": "hebei_station00", "pred_len": 16},
            {"model": "TextFiLM", "phase": "text_stage2", "station": "hebei_station00", "pred_len": 16},
        ]
        args.epochs = min(args.epochs, 1)
        gpus = gpus[:1]
    print(f"Launching {len(tasks)} tasks on GPUs {gpus}: models={models} phases={phases} stations={len(stations)} pred_lens={pred_lens}", flush=True)
    q = Queue()
    for task in tasks:
        q.put(task)
    for _ in gpus:
        q.put(None)
    procs = [Process(target=worker, args=(gpu, q, args)) for gpu in gpus]
    for p in procs:
        p.start()
    try:
        while any(p.is_alive() for p in procs):
            write_summary()
            time.sleep(60)
    finally:
        for p in procs:
            p.join()
    write_summary()


if __name__ == "__main__":
    main()
