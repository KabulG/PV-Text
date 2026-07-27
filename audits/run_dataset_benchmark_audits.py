import csv
import json
import math
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/root/KDD")
DATA = ROOT / "hebei_daytime_0800_1900"
OUT = ROOT / "audits_pv_text_benchmark"
OUT.mkdir(parents=True, exist_ok=True)

STATIONS = [f"hebei_station{i:02d}" for i in range(10)]
PRED_LENS = [16, 32, 48, 96]
SEQ_LEN = 96

TEXT1_FIELDS = ["state_prompt", "recent_trend_prompt", "statistical_variability_prompt"]
TEXT2_FIELDS = ["low_frequency_trend_prompt", "high_frequency_component_prompt"]


def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.floor("h")
    return df


def tokens(text):
    return re.findall(r"[A-Za-z0-9_]+", str(text).lower())


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def classify_trend(text):
    s = str(text).lower()
    if any(x in s for x in ["increase", "increasing", "rise", "rising", "upward", "climb"]):
        return "increase"
    if any(x in s for x in ["decrease", "decreasing", "fall", "falling", "drop", "downward", "decline"]):
        return "decrease"
    if any(x in s for x in ["steady", "stable", "little net change", "broadly stable"]):
        return "steady"
    return "unknown"


def classify_level(text):
    s = str(text).lower()
    if any(x in s for x in ["very weak", "low", "smooth", "weak"]):
        return "low"
    if any(x in s for x in ["high", "strong", "large", "obvious", "mutation-like jumps"]):
        return "high"
    if any(x in s for x in ["moderate", "medium"]):
        return "medium"
    return "unknown"


def numeric_trend(delta, scale):
    thr = max(scale * 0.02, 1e-6)
    if delta > thr:
        return "increase"
    if delta < -thr:
        return "decrease"
    return "steady"


def split_borders(n):
    n_train = int(n * 0.7)
    n_test = int(n * 0.2)
    n_val = n - n_train - n_test
    return [0, n_train - SEQ_LEN, n - n_test - SEQ_LEN], [n_train, n_train + n_val, n]


def audit_text_statistics():
    rows = []
    vocab = defaultdict(Counter)
    for station in STATIONS:
        solar = pd.read_csv(DATA / station / "solar.csv")
        solar_dates = set(pd.to_datetime(solar["date"]).dt.floor("h"))
        for stage, fname, fields in [
            ("text_stage1", "rt_text1.jsonl", TEXT1_FIELDS),
            ("text_stage2", "rt_text2.jsonl", TEXT2_FIELDS),
        ]:
            df = read_jsonl(DATA / station / fname)
            ts = set(df["timestamp"]) if not df.empty else set()
            base = {
                "station": station,
                "stage": stage,
                "rows": len(df),
                "solar_rows": len(solar),
                "timestamp_coverage": len(ts & solar_dates) / max(len(solar_dates), 1),
                "extra_text_timestamps": len(ts - solar_dates),
                "missing_text_timestamps": len(solar_dates - ts),
                "duplicate_timestamps": int(df["timestamp"].duplicated().sum()) if not df.empty else 0,
            }
            for field in fields:
                vals = df[field].fillna("").astype(str) if field in df else pd.Series([], dtype=str)
                nonempty = vals.str.strip().ne("").sum()
                lens_char = vals.map(len)
                lens_tok = vals.map(lambda x: len(tokens(x)))
                unique = vals.nunique(dropna=False)
                all_toks = []
                for v in vals:
                    all_toks.extend(tokens(v))
                vocab[(station, stage, field)].update(all_toks)
                row = dict(base)
                row.update({
                    "field": field,
                    "nonempty": int(nonempty),
                    "missing": int(len(vals) - nonempty),
                    "coverage": float(nonempty / max(len(vals), 1)),
                    "avg_chars": float(lens_char.mean()) if len(vals) else 0.0,
                    "avg_tokens": float(lens_tok.mean()) if len(vals) else 0.0,
                    "unique_texts": int(unique),
                    "duplicate_rate": float(1 - unique / max(len(vals), 1)),
                    "vocab_size": len(set(all_toks)),
                })
                rows.append(row)
    write_csv(OUT / "text_quality_statistics.csv", rows, list(rows[0].keys()))
    return rows


def audit_leakage_and_alignment():
    rows = []
    risky_rows = []
    risky_patterns = ["coming horizon", "future", "next hour", "next", "forecast", "ahead", "prediction horizon"]
    for station in STATIONS:
        raw = pd.read_csv(DATA / station / "solar.csv")
        dates = pd.to_datetime(raw["date"]).dt.floor("h").reset_index(drop=True)
        n = len(raw)
        border1s, border2s = split_borders(n)
        for phase, fname, fields in [
            ("text_stage1", "rt_text1.jsonl", TEXT1_FIELDS),
            ("text_stage2", "rt_text2.jsonl", TEXT2_FIELDS),
        ]:
            df = read_jsonl(DATA / station / fname)
            df["_text"] = df[fields].fillna("").astype(str).agg(" ".join, axis=1)
            text_ts = set(df["timestamp"])
            for _, r in df.iterrows():
                lower = r["_text"].lower()
                hits = [p for p in risky_patterns if p in lower]
                if hits:
                    risky_rows.append({
                        "station": station,
                        "phase": phase,
                        "timestamp": str(r["timestamp"]),
                        "patterns": ";".join(hits),
                        "text": r["_text"][:500],
                    })
            for pred_len in PRED_LENS:
                total = 0
                code_future_leak = 0
                same_hour_as_first_prediction = 0
                missing = 0
                for set_type in range(3):
                    b1, b2 = border1s[set_type], border2s[set_type]
                    max_i = b2 - b1 - SEQ_LEN - pred_len + 1
                    for idx in range(max(0, max_i)):
                        s = b1 + idx
                        e = s + SEQ_LEN
                        y0 = e
                        text_time = dates.iloc[e - 1]
                        last_observed_actual = pd.to_datetime(raw["date"]).iloc[e - 1]
                        pred_start_actual = pd.to_datetime(raw["date"]).iloc[y0]
                        pred_start_hour = dates.iloc[y0]
                        total += 1
                        # This mirrors the model code: it chooses dates[e-1].floor("h").
                        # A true code-level future leak only occurs if the selected text hour
                        # is after the last observed timestamp's hour.
                        if text_time > pd.Timestamp(last_observed_actual).floor("h"):
                            code_future_leak += 1
                        # Separate boundary-risk statistic for 15-min data: the selected text
                        # hour can equal the first prediction's hour after flooring.
                        if text_time == pred_start_hour:
                            same_hour_as_first_prediction += 1
                        if text_time not in text_ts:
                            missing += 1
                rows.append({
                    "station": station,
                    "phase": phase,
                    "pred_len": pred_len,
                    "samples": total,
                    "code_future_text_leak_count": code_future_leak,
                    "same_hour_bucket_as_first_prediction": same_hour_as_first_prediction,
                    "missing_last_observed_text": missing,
                    "code_future_text_leak_rate": code_future_leak / max(total, 1),
                    "same_hour_bucket_rate": same_hour_as_first_prediction / max(total, 1),
                })
    write_csv(OUT / "timestamp_leakage_audit.csv", rows, list(rows[0].keys()))
    write_csv(OUT / "content_future_phrase_audit.csv", risky_rows, ["station", "phase", "timestamp", "patterns", "text"])
    return rows, risky_rows


def audit_representativeness():
    rows = []
    examples = []
    for station in STATIONS:
        raw = pd.read_csv(DATA / station / "solar.csv")
        raw["date"] = pd.to_datetime(raw["date"]).dt.floor("h")
        target = raw["OT"].astype(float).values
        scale = float(np.nanmax(target) - np.nanmin(target))
        delta = np.r_[0.0, np.diff(target)]
        trend_num = [numeric_trend(d, scale) for d in delta]
        roll_std = pd.Series(target).rolling(6, min_periods=2).std().fillna(0).values
        q1, q2 = np.quantile(roll_std, [0.33, 0.67])
        var_num = np.where(roll_std <= q1, "low", np.where(roll_std >= q2, "high", "medium"))
        near_zero_thr = max(scale * 0.02, 1e-6)
        near_zero = target <= near_zero_thr

        df1 = read_jsonl(DATA / station / "rt_text1.jsonl")
        merged1 = raw[["date", "OT"]].merge(df1, left_on="date", right_on="timestamp", how="inner")
        idx_map = {t: i for i, t in enumerate(raw["date"])}
        state_total = state_ok = 0
        trend_total = trend_ok = 0
        var_total = var_ok = 0
        for _, r in merged1.iterrows():
            i = idx_map[r["date"]]
            state = str(r.get("state_prompt", "")).lower()
            if "near zero" in state or "weak radiation" in state or "nighttime" in state:
                state_total += 1
                ok = bool(near_zero[i])
                state_ok += int(ok)
                if not ok and len(examples) < 50:
                    examples.append({"station": station, "timestamp": str(r["date"]), "check": "state_near_zero", "numeric": float(target[i]), "text": state[:300]})
            tr = classify_trend(r.get("recent_trend_prompt", ""))
            if tr != "unknown":
                trend_total += 1
                ok = tr == trend_num[i]
                trend_ok += int(ok)
                if not ok and len(examples) < 50:
                    examples.append({"station": station, "timestamp": str(r["date"]), "check": "recent_trend", "numeric": trend_num[i], "text": str(r.get("recent_trend_prompt", ""))[:300]})
            vl = classify_level(r.get("statistical_variability_prompt", ""))
            if vl != "unknown":
                var_total += 1
                ok = vl == var_num[i]
                var_ok += int(ok)
                if not ok and len(examples) < 50:
                    examples.append({"station": station, "timestamp": str(r["date"]), "check": "variability", "numeric": str(var_num[i]), "text": str(r.get("statistical_variability_prompt", ""))[:300]})

        rows.extend([
            {"station": station, "check": "stage1_state_near_zero", "total": state_total, "agree": state_ok, "agreement": state_ok / max(state_total, 1)},
            {"station": station, "check": "stage1_recent_trend", "total": trend_total, "agree": trend_ok, "agreement": trend_ok / max(trend_total, 1)},
            {"station": station, "check": "stage1_variability_level", "total": var_total, "agree": var_ok, "agreement": var_ok / max(var_total, 1)},
        ])

        df2 = read_jsonl(DATA / station / "rt_text2.jsonl")
        merged2 = raw[["date", "OT"]].merge(df2, left_on="date", right_on="timestamp", how="inner")
        high_total = high_ok = low_total = low_ok = 0
        smooth = pd.Series(target).rolling(12, min_periods=2).mean().bfill().ffill().values
        residual = np.abs(target - smooth)
        rq1, rq2 = np.quantile(residual, [0.33, 0.67])
        comp_num = np.where(residual <= rq1, "low", np.where(residual >= rq2, "high", "medium"))
        for _, r in merged2.iterrows():
            i = idx_map[r["date"]]
            hf = classify_level(r.get("high_frequency_component_prompt", ""))
            if hf != "unknown":
                high_total += 1
                ok = hf == comp_num[i]
                high_ok += int(ok)
            lf = str(r.get("low_frequency_trend_prompt", "")).lower()
            if "nighttime baseline" in lf or "near a nighttime baseline" in lf:
                low_total += 1
                ok = bool(near_zero[i])
                low_ok += int(ok)
        rows.extend([
            {"station": station, "check": "stage2_high_frequency_level", "total": high_total, "agree": high_ok, "agreement": high_ok / max(high_total, 1)},
            {"station": station, "check": "stage2_lowfreq_night_baseline", "total": low_total, "agree": low_ok, "agreement": low_ok / max(low_total, 1)},
        ])
    write_csv(OUT / "text_numeric_representativeness_audit.csv", rows, ["station", "check", "total", "agree", "agreement"])
    write_csv(OUT / "representativeness_mismatch_examples.csv", examples, ["station", "timestamp", "check", "numeric", "text"])
    return rows


def audit_results_and_fairness():
    rows = []
    ts_path = ROOT / "final_result_tables/summary_all_models_hebei_ts.csv"
    text_path = ROOT / "experiments_text_fusion_models/summary_text_fusion_aligned_16_32_48_96.csv"
    onehot_path = ROOT / "experiments_iTransformer_hebei_text_full/summary_text_full.csv"
    ts = pd.read_csv(ts_path)
    text = pd.read_csv(text_path)
    onehot = pd.read_csv(onehot_path)
    rows.append({"check": "pure_ts_rows", "value": len(ts), "expected": 7 * 10 * 4, "pass": len(ts) == 7 * 10 * 4})
    rows.append({"check": "text_fusion_rows", "value": len(text), "expected": 4 * 3 * 10 * 4, "pass": len(text) == 4 * 3 * 10 * 4})
    rows.append({"check": "text_fusion_failures", "value": int((text["status"] != "success").sum()), "expected": 0, "pass": int((text["status"] != "success").sum()) == 0})
    rows.append({"check": "onehot_rows", "value": len(onehot), "expected": 4 * 10 * 4, "pass": len(onehot) == 4 * 10 * 4})
    ts_preds = sorted(ts["pred_len"].unique().tolist())
    text_preds = sorted(text["pred_len"].unique().tolist())
    rows.append({"check": "pure_ts_pred_lens", "value": str(ts_preds), "expected": str(PRED_LENS), "pass": ts_preds == PRED_LENS})
    rows.append({"check": "text_pred_lens", "value": str(text_preds), "expected": str(PRED_LENS), "pass": text_preds == PRED_LENS})
    by_model = ts.groupby("model")["target_mae"].mean().sort_values()
    if len(by_model) >= 2:
        best = by_model.index[0]
        second = by_model.index[1]
        ratio = float(by_model.iloc[1] / max(by_model.iloc[0], 1e-12))
        rows.append({"check": "pure_ts_best_model_mae_gap", "value": f"{best}={by_model.iloc[0]:.6f}, {second}={by_model.iloc[1]:.6f}, ratio={ratio:.2f}", "expected": "manual_audit_if_ratio_gt_3", "pass": ratio <= 3})
    # File/log existence
    missing_logs = 0
    for _, r in ts.iterrows():
        if "log" in r and isinstance(r["log"], str) and r["log"] and not Path(r["log"]).exists():
            missing_logs += 1
    rows.append({"check": "pure_ts_missing_logs", "value": missing_logs, "expected": 0, "pass": missing_logs == 0})
    write_csv(OUT / "result_fairness_audit.csv", rows, ["check", "value", "expected", "pass"])
    return rows


def write_report(text_stats, leak_rows, risky_rows, repr_rows, fair_rows):
    rep = []
    rep.append("# PV-Text Dataset and Benchmark Audit Report\n")
    rep.append("## Execution Status\n")
    rep.append("```text\n")
    rep.append(subprocess.getoutput("ps -ef | grep -E 'run_text_fusion_benchmark|run_hebei|run_tsl' | grep -v grep || true"))
    rep.append("\n")
    rep.append(subprocess.getoutput("nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader"))
    rep.append("\n```\n")
    rep.append("## Key Findings\n")
    fair_fail = [r for r in fair_rows if str(r["pass"]) != "True"]
    rep.append(f"- Text quality rows audited: {len(text_stats)} field-level station-stage rows.\n")
    rep.append(f"- Timestamp leakage audit rows: {len(leak_rows)}; total code-level future text leak count: {sum(int(r['code_future_text_leak_count']) for r in leak_rows)}.\n")
    rep.append(f"- Same-hour boundary cases after hourly flooring: {sum(int(r['same_hour_bucket_as_first_prediction']) for r in leak_rows)}. These require wording in the paper because the numerical data are sub-hourly while text is hourly.\n")
    rep.append(f"- Potential future-looking phrase rows: {len(risky_rows)}. These are content-level wording flags, not model-input timestamp leakage.\n")
    rep.append(f"- Representativeness checks: {len(repr_rows)} station-check rows.\n")
    rep.append(f"- Result/fairness audit failed or warning checks: {len(fair_fail)}.\n")
    if fair_fail:
        rep.append("\n## Warnings\n")
        for r in fair_fail:
            rep.append(f"- {r['check']}: value={r['value']}, expected={r['expected']}.\n")
    rep.append("\n## Files\n")
    for name in [
        "text_quality_statistics.csv",
        "timestamp_leakage_audit.csv",
        "content_future_phrase_audit.csv",
        "text_numeric_representativeness_audit.csv",
        "representativeness_mismatch_examples.csv",
        "result_fairness_audit.csv",
    ]:
        rep.append(f"- `{OUT / name}`\n")
    (OUT / "AUDIT_REPORT.md").write_text("".join(rep), encoding="utf-8")


def main():
    print("Running text statistics...")
    text_stats = audit_text_statistics()
    print("Running leakage audit...")
    leak_rows, risky_rows = audit_leakage_and_alignment()
    print("Running representativeness audit...")
    repr_rows = audit_representativeness()
    print("Running result/fairness audit...")
    fair_rows = audit_results_and_fairness()
    write_report(text_stats, leak_rows, risky_rows, repr_rows, fair_rows)
    print("AUDIT_DIR", OUT)
    print((OUT / "AUDIT_REPORT.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
