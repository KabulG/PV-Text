#!/usr/bin/env python3
"""Fail if a data directory contains stations outside the authorized Hebei scope."""

from __future__ import annotations

import argparse
from pathlib import Path


ALLOWED = {f"hebei_station{i:02d}" for i in range(10)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, default=Path("data/hebei_daytime_0800_1900"))
    args = parser.parse_args()
    stations = sorted(p.name for p in args.data_root.iterdir() if p.is_dir())
    unauthorized = [station for station in stations if station not in ALLOWED]
    missing = sorted(ALLOWED - set(stations))
    if unauthorized or missing:
        print("Authorized scope check failed.")
        if unauthorized:
            print("Unauthorized stations:", ", ".join(unauthorized))
        if missing:
            print("Missing Hebei stations:", ", ".join(missing))
        return 1
    print("Authorized scope check passed: hebei_station00-hebei_station09 only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
