"""
Generate PV-Text-style hourly PV prompt JSONL files for multiple stations.

This script follows the existing two-file prompt contract:
- *_stage1_hourly_prompts.jsonl:
  timestamp, state_prompt, recent_trend_prompt, statistical_variability_prompt
- *_stage2_hourly_prompts.jsonl:
  timestamp, low_frequency_trend_prompt, high_frequency_component_prompt

Default column schema is based on datasets/station00/source_station.csv.
"""

from __future__ import annotations

import argparse
import glob
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PVColumns:
    date: str = "date"
    temperature: str = "temp"
    pressure: str = "pressure"
    wind_speed: str = "wind_speed"
    wind_dir: str = "wind_dir"
    radiation: str = "sun_radiation"
    diffuse_radiation: str = "sca_radiation"
    nwp_radiation: str = "nwp_shortwaveirrad"
    nwp_diffuse_radiation: str = "nwp_scatterirrad"
    capacity: str = "capacity"
    target: str = "OT"


@dataclass(frozen=True)
class StationStats:
    radiation_q20: float
    radiation_q50: float
    radiation_q75: float
    nwp_q20: float
    nwp_q50: float
    nwp_q75: float
    temp_q25: float
    power_ref: float
    power_flat_eps: float
    radiation_flat_eps: float
    nwp_flat_eps: float
    variability_q40: float
    variability_q75: float
    hf_q40: float
    hf_q70: float
    hf_q88: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export PV-Text-style hourly prompt jsonl files.")
    parser.add_argument("--input", type=str, default="datasets/station00/source_station.csv")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--station_id", type=str, default="")
    parser.add_argument("--future_hours", type=int, default=6)
    parser.add_argument("--max_hours", type=int, default=-1)
    parser.add_argument("--start_hour", type=int, default=0)
    parser.add_argument("--date_col", type=str, default="date")
    parser.add_argument("--temperature_col", type=str, default="temp")
    parser.add_argument("--pressure_col", type=str, default="pressure")
    parser.add_argument("--wind_speed_col", type=str, default="wind_speed")
    parser.add_argument("--wind_dir_col", type=str, default="wind_dir")
    parser.add_argument("--radiation_col", type=str, default="sun_radiation")
    parser.add_argument("--diffuse_radiation_col", type=str, default="sca_radiation")
    parser.add_argument("--nwp_radiation_col", type=str, default="nwp_shortwaveirrad")
    parser.add_argument("--nwp_diffuse_radiation_col", type=str, default="nwp_scatterirrad")
    parser.add_argument("--capacity_col", type=str, default="capacity")
    parser.add_argument("--target_col", type=str, default="OT")
    return parser.parse_args()


def iter_csv_files(pattern: str) -> Iterable[Path]:
    matches = sorted(Path(p) for p in glob.glob(pattern))
    if matches:
        return matches
    path = Path(pattern)
    if path.exists():
        return [path]
    raise FileNotFoundError(f"No CSV files matched: {pattern}")


def columns_from_args(args: argparse.Namespace) -> PVColumns:
    return PVColumns(
        date=args.date_col,
        temperature=args.temperature_col,
        pressure=args.pressure_col,
        wind_speed=args.wind_speed_col,
        wind_dir=args.wind_dir_col,
        radiation=args.radiation_col,
        diffuse_radiation=args.diffuse_radiation_col,
        nwp_radiation=args.nwp_radiation_col,
        nwp_diffuse_radiation=args.nwp_diffuse_radiation_col,
        capacity=args.capacity_col,
        target=args.target_col,
    )


def load_station_frame(csv_path: Path, cols: PVColumns) -> pd.DataFrame:
    raw = pd.read_csv(csv_path)
    required = [
        cols.date,
        cols.temperature,
        cols.wind_speed,
        cols.radiation,
        cols.diffuse_radiation,
        cols.capacity,
        cols.target,
    ]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f"{csv_path} missing required columns: {missing}")

    df = pd.DataFrame()
    df["date"] = pd.to_datetime(raw[cols.date])
    df["temp"] = pd.to_numeric(raw[cols.temperature], errors="coerce")
    df["wind_speed"] = pd.to_numeric(raw[cols.wind_speed], errors="coerce")
    df["radiation"] = pd.to_numeric(raw[cols.radiation], errors="coerce")
    df["diffuse_radiation"] = pd.to_numeric(raw[cols.diffuse_radiation], errors="coerce")
    df["capacity"] = pd.to_numeric(raw[cols.capacity], errors="coerce")
    df["power"] = pd.to_numeric(raw[cols.target], errors="coerce")

    if cols.nwp_radiation in raw.columns:
        df["nwp_radiation"] = pd.to_numeric(raw[cols.nwp_radiation], errors="coerce")
    else:
        df["nwp_radiation"] = df["radiation"]

    if cols.nwp_diffuse_radiation in raw.columns:
        df["nwp_diffuse_radiation"] = pd.to_numeric(raw[cols.nwp_diffuse_radiation], errors="coerce")
    else:
        df["nwp_diffuse_radiation"] = df["diffuse_radiation"]

    df = df.sort_values("date").reset_index(drop=True)
    value_cols = [c for c in df.columns if c != "date"]
    df[value_cols] = df[value_cols].interpolate(limit_direction="both").fillna(0.0)
    return df


def hourly_frame(df: pd.DataFrame) -> pd.DataFrame:
    data = df.set_index("date").resample("1h").agg(
        {
            "temp": "mean",
            "wind_speed": "mean",
            "radiation": ["mean", "std", "first", "last", "max"],
            "diffuse_radiation": "mean",
            "nwp_radiation": ["mean", "first", "last"],
            "nwp_diffuse_radiation": "mean",
            "capacity": "max",
            "power": ["mean", "std", "first", "last", "max"],
        }
    )
    data.columns = [
        "temp_mean",
        "wind_speed_mean",
        "radiation_mean",
        "radiation_std",
        "radiation_first",
        "radiation_last",
        "radiation_max",
        "diffuse_mean",
        "nwp_mean",
        "nwp_first",
        "nwp_last",
        "nwp_diffuse_mean",
        "capacity_max",
        "power_mean",
        "power_std",
        "power_first",
        "power_last",
        "power_max",
    ]
    data = data.dropna(subset=["radiation_mean", "power_mean"]).reset_index()
    data["radiation_std"] = data["radiation_std"].fillna(0.0)
    data["power_std"] = data["power_std"].fillna(0.0)
    return data


def build_stats(hourly: pd.DataFrame) -> StationStats:
    daylight = hourly[hourly["radiation_mean"] > 0]
    if daylight.empty:
        daylight = hourly
    active_power = hourly[hourly["power_mean"] > 0]["power_mean"]
    if active_power.empty:
        active_power = hourly["power_mean"]

    capacity = hourly["capacity_max"].replace([np.inf, -np.inf], np.nan).dropna()
    power_ref = float(capacity.max()) if len(capacity) and capacity.max() > 0 else float(np.nanpercentile(active_power, 95))
    power_ref = max(power_ref, 1e-6)

    radiation_ref = max(float(np.nanpercentile(daylight["radiation_mean"], 95)), 1e-6)
    nwp_ref = max(float(np.nanpercentile(daylight["nwp_mean"], 95)), 1e-6)
    variability = hourly["power_std"] / power_ref
    hf = (hourly["power_std"] / power_ref) + 0.5 * (hourly["radiation_std"] / radiation_ref)

    return StationStats(
        radiation_q20=float(np.nanpercentile(daylight["radiation_mean"], 20)),
        radiation_q50=float(np.nanpercentile(daylight["radiation_mean"], 50)),
        radiation_q75=float(np.nanpercentile(daylight["radiation_mean"], 75)),
        nwp_q20=float(np.nanpercentile(daylight["nwp_mean"], 20)),
        nwp_q50=float(np.nanpercentile(daylight["nwp_mean"], 50)),
        nwp_q75=float(np.nanpercentile(daylight["nwp_mean"], 75)),
        temp_q25=float(np.nanpercentile(hourly["temp_mean"], 25)),
        power_ref=power_ref,
        power_flat_eps=max(0.02 * power_ref, 1e-6),
        radiation_flat_eps=max(0.03 * radiation_ref, 1e-6),
        nwp_flat_eps=max(0.03 * nwp_ref, 1e-6),
        variability_q40=float(np.nanpercentile(variability, 40)),
        variability_q75=float(np.nanpercentile(variability, 75)),
        hf_q40=float(np.nanpercentile(hf, 40)),
        hf_q70=float(np.nanpercentile(hf, 70)),
        hf_q88=float(np.nanpercentile(hf, 88)),
    )


def trend_label(delta: float, eps: float, up: str, down: str, flat: str) -> str:
    if delta > eps:
        return up
    if delta < -eps:
        return down
    return flat


def state_prompt(row: pd.Series, stats: StationStats) -> str:
    rad = row["radiation_mean"]
    power = row["power_mean"]
    if rad <= stats.radiation_q20 or power <= 0.02 * stats.power_ref:
        text = "The station is under nighttime or very weak radiation conditions, and photovoltaic output remains near zero."
    elif rad <= stats.radiation_q50:
        text = "The station is under weak radiation conditions, and photovoltaic output is still suppressed."
    elif rad <= stats.radiation_q75:
        text = "The station is under moderate radiation conditions, and photovoltaic output is in an active but not peak-producing state."
    else:
        text = "The station is under strong radiation conditions, and photovoltaic output is in an active daytime regime."

    modifiers = []
    if row["temp_mean"] <= stats.temp_q25:
        modifiers.append("The thermal background is relatively cool")
    if row["wind_speed_mean"] <= 1.5:
        modifiers.append("wind is light")
    if row["nwp_mean"] > stats.nwp_q20 and rad > stats.radiation_q20:
        modifiers.append("the NWP radiation background is available for guidance")

    if modifiers:
        text += " " + "; ".join(modifiers) + "."
    return text


def recent_trend_prompt(row: pd.Series, stats: StationStats) -> str:
    output = trend_label(
        row["power_last"] - row["power_first"],
        stats.power_flat_eps,
        "Output has been recovering over the recent hour.",
        "Output has softened over the recent hour.",
        "Output has stayed broadly steady over the recent hour.",
    )
    radiation = trend_label(
        row["radiation_last"] - row["radiation_first"],
        stats.radiation_flat_eps,
        "Observed radiation has strengthened.",
        "Observed radiation has weakened.",
        "Observed radiation has stayed broadly stable.",
    )
    nwp = trend_label(
        row["nwp_last"] - row["nwp_first"],
        stats.nwp_flat_eps,
        "The NWP shortwave background is also trending upward.",
        "The NWP shortwave background is also trending downward.",
        "The NWP shortwave background shows little net change.",
    )
    return f"{output} {radiation} {nwp}"


def variability_prompt(row: pd.Series, stats: StationStats) -> str:
    score = row["power_std"] / stats.power_ref
    if score <= stats.variability_q40:
        return "Recent variability is low, and the series remains locally smooth."
    if score <= stats.variability_q75:
        return "Recent variability is moderate, with some short-term fluctuation but no severe instability."
    return "Recent variability is high, and the series shows clear short-term disturbance."


def low_frequency_prompt(hourly: pd.DataFrame, idx: int, stats: StationStats, future_hours: int) -> str:
    future = hourly.iloc[idx : min(idx + future_hours, len(hourly))]
    if future.empty:
        future = hourly.iloc[[idx]]

    start = float(future.iloc[0]["nwp_mean"])
    end = float(future.iloc[-1]["nwp_mean"])
    avg = float(future["nwp_mean"].mean())
    diffuse_ratio = float(future["nwp_diffuse_mean"].mean()) / max(avg, 1e-6)

    if avg <= stats.nwp_q20:
        return "The low-frequency background remains near a nighttime baseline through the coming horizon."

    if end - start > stats.nwp_flat_eps:
        prefix = "The low-frequency background is expected to strengthen gradually"
        join = "with the radiation regime leaning toward"
    elif end - start < -stats.nwp_flat_eps:
        prefix = "The low-frequency background is expected to weaken gradually"
        join = "with the radiation regime characterized by"
    else:
        prefix = "The low-frequency background is expected to stay broadly stable"
        join = "under"

    if diffuse_ratio < 0.35:
        regime = "a direct-beam-dominant background"
    elif diffuse_ratio < 0.75:
        regime = "a mixed direct and diffuse background"
    else:
        regime = "a diffuse-cloud-dominant background"
    return f"{prefix}, {join} {regime}."


def high_frequency_prompt(row: pd.Series, stats: StationStats) -> str:
    radiation_component = row["radiation_std"] / max(stats.radiation_q75, 1e-6)
    power_component = row["power_std"] / stats.power_ref
    score = power_component + 0.5 * radiation_component
    if score <= stats.hf_q40:
        return "The high-frequency component is very weak, and obvious mutation-like jumps are unlikely."
    if score <= stats.hf_q70:
        return "The high-frequency component remains limited, with only mild short-lived fluctuation expected."
    if score <= stats.hf_q88:
        return "The high-frequency component may contain intermittent cloud-driven jumps and short-lived reversals."
    return "The high-frequency component is active, and sharp cloud-edge-like jumps or rapid reversals are likely."


def export_station(csv_path: Path, args: argparse.Namespace) -> tuple[Path, Path]:
    cols = columns_from_args(args)
    df = load_station_frame(csv_path, cols)
    hourly = hourly_frame(df)
    stats = build_stats(hourly)

    station_id = args.station_id or csv_path.stem
    output_dir = Path(args.output_dir) if args.output_dir else csv_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    stage1_path = output_dir / f"{station_id}_stage1_hourly_prompts.jsonl"
    stage2_path = output_dir / f"{station_id}_stage2_hourly_prompts.jsonl"

    max_len = max(0, len(hourly) - max(args.future_hours, 1))
    start = max(args.start_hour, 0)
    end = max_len if args.max_hours < 0 else min(max_len, start + args.max_hours)

    with stage1_path.open("w", encoding="utf-8") as f1, stage2_path.open("w", encoding="utf-8") as f2:
        for idx in range(start, end):
            row = hourly.iloc[idx]
            timestamp = pd.Timestamp(row["date"]).strftime("%Y-%m-%d %H:%M:%S")
            stage1 = {
                "timestamp": timestamp,
                "state_prompt": state_prompt(row, stats),
                "recent_trend_prompt": recent_trend_prompt(row, stats),
                "statistical_variability_prompt": variability_prompt(row, stats),
            }
            stage2 = {
                "timestamp": timestamp,
                "low_frequency_trend_prompt": low_frequency_prompt(hourly, idx, stats, args.future_hours),
                "high_frequency_component_prompt": high_frequency_prompt(row, stats),
            }
            f1.write(json.dumps(stage1, ensure_ascii=False) + "\n")
            f2.write(json.dumps(stage2, ensure_ascii=False) + "\n")

    print(f"[{station_id}] wrote {end - start} hourly records")
    print(f"  {stage1_path}")
    print(f"  {stage2_path}")
    return stage1_path, stage2_path


def main() -> None:
    args = parse_args()
    for csv_path in iter_csv_files(args.input):
        export_station(csv_path, args)


if __name__ == "__main__":
    main()
