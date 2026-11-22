#!/usr/bin/env python3
"""Basic integrity checks for locally accumulated FlowFirst JSONL data."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path("data/flowfirst")
REQUIRED = {
    "taker": {"symbol", "period", "timestamp", "buyVol", "sellVol"},
    "oi": {"symbol", "period", "timestamp", "sumOpenInterest", "sumOpenInterestValue"},
    "klines": {"symbol", "period", "timestamp", "close_time", "open", "high", "low", "close"},
}


def main() -> None:
    if not ROOT.exists():
        raise SystemExit("No data/flowfirst directory found.")

    totals = Counter()
    seen = set()
    errors = []
    files = sorted(ROOT.glob("*.jsonl"))
    for path in files:
        parts = path.stem.split("_")
        if len(parts) < 3:
            errors.append(f"{path}: unexpected filename")
            continue
        kind, period = parts[-2], parts[-1]
        if kind not in REQUIRED:
            errors.append(f"{path}: unknown data kind {kind}")
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                try:
                    row = json.loads(line)
                    missing = REQUIRED[kind] - row.keys()
                    if missing:
                        raise ValueError(f"missing fields {sorted(missing)}")
                    if row["period"] != period:
                        raise ValueError("period does not match filename")
                    key = (kind, row["symbol"], row["period"], int(row["timestamp"]))
                    if key in seen:
                        raise ValueError("duplicate symbol/period/timestamp")
                    seen.add(key)
                    totals[(kind, row["symbol"], row["period"])] += 1
                except Exception as exc:
                    errors.append(f"{path}:{line_no}: {exc}")

    print(f"Files checked: {len(files)}")
    print(f"Unique records checked: {len(seen)}")
    for key, count in sorted(totals.items()):
        print(f"{key[0]:7s} {key[1]:10s} {key[2]:3s}: {count}")
    if errors:
        for error in errors[:50]:
            print(f"ERROR: {error}")
        raise SystemExit(f"Integrity validation failed: {len(errors)} issue(s).")
    if not files:
        raise SystemExit("No JSONL data files found.")
    print("FlowFirst raw-data integrity checks passed.")


if __name__ == "__main__":
    main()
