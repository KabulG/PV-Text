#!/usr/bin/env python3
"""Create deterministic chronological splits for PV-Text station files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


STATIONS = [f"hebei_station{i:02d}" for i in range(10)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, default=Path("data/hebei_daytime_0800_1900"))
    parser.add_argument("--output", type=Path, default=Path("splits/hebei_chronological_splits.csv"))
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for station in STATIONS:
        csv_path = args.data_root / station / "solar.csv"
        df = pd.read_csv(csv_path)
        n = len(df)
        n_train = int(n * args.train_ratio)
        n_test = int(n * args.test_ratio)
        n_val = n - n_train - n_test
        cuts = {
            "train": (0, n_train),
            "validation": (n_train, n_train + n_val),
            "test": (n_train + n_val, n),
        }
        for split, (start, end) in cuts.items():
            rows.append(
                {
                    "station": station,
                    "split": split,
                    "start_index": start,
                    "end_index_exclusive": end,
                    "rows": end - start,
                    "start_time": df["date"].iloc[start],
                    "end_time": df["date"].iloc[end - 1],
                }
            )
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
