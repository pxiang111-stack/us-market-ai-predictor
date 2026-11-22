#!/usr/bin/env python3
"""FlowFirst data coverage report: counts, ranges, gaps and freshness by feed."""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("data/flowfirst")
OUT = Path("reports")
INTERVAL_MS = {"5m": 300_000, "15m": 900_000, "30m": 1_800_000,
               "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000,
               "1d": 86_400_000}


def main() -> None:
    groups: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    files = sorted(ROOT.glob("*.jsonl")) if ROOT.exists() else []
    for path in files:
        parts = path.stem.split("_")
        if len(parts) < 3:
            continue
        kind, period = parts[-2], parts[-1]
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                    groups[(kind, row["symbol"], period)].append(int(row["timestamp"]))
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    continue

    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    feeds = []
    for (kind, symbol, period), stamps in sorted(groups.items()):
        stamps = sorted(set(stamps))
        interval = INTERVAL_MS.get(period)
        gaps = 0
        largest_gap_bars = 0
        if interval and len(stamps) > 1:
            for left, right in zip(stamps, stamps[1:]):
                missing = max(0, round((right - left) / interval) - 1)
                gaps += missing
                largest_gap_bars = max(largest_gap_bars, missing)
        latest_age = round(max(0, now_ms - stamps[-1]) / 60_000, 1) if stamps else None
        feeds.append({
            "kind": kind, "symbol": symbol, "period": period,
            "records": len(stamps),
            "start_utc": datetime.fromtimestamp(stamps[0] / 1000, timezone.utc).isoformat() if stamps else None,
            "end_utc": datetime.fromtimestamp(stamps[-1] / 1000, timezone.utc).isoformat() if stamps else None,
            "latest_age_minutes": latest_age,
            "estimated_missing_intervals": gaps,
            "largest_gap_intervals": largest_gap_bars,
            "expected_interval_minutes": interval / 60_000 if interval else None,
        })

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "local accumulated JSONL snapshot; not a live-feed guarantee",
        "file_count": len(files),
        "feed_count": len(feeds),
        "total_unique_records": sum(x["records"] for x in feeds),
        "feeds": feeds,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "flowfirst_data_coverage.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# FlowFirst data coverage",
        "",
        f"Generated UTC: {report['generated_at_utc']}",
        f"Files: {len(files)} | Feeds: {len(feeds)} | Unique records across feeds: {report['total_unique_records']}",
        "",
        "Missing intervals are estimated from timestamp spacing; they are not proof that the exchange published every expected interval.",
        "",
        "| Feed | Symbol | Period | Records | First UTC | Latest UTC | Age (min) | Est. missing bars | Largest gap (bars) |",
        "|---|---|---:|---:|---|---|---:|---:|---:|",
    ]
    for x in feeds:
        lines.append("| " + " | ".join([
            x["kind"], x["symbol"], x["period"], str(x["records"]),
            x["start_utc"] or "-", x["end_utc"] or "-",
            str(x["latest_age_minutes"]), str(x["estimated_missing_intervals"]),
            str(x["largest_gap_intervals"]),
        ]) + " |")
    (OUT / "flowfirst_data_coverage.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "file_count": report["file_count"],
        "feed_count": report["feed_count"],
        "total_unique_records": report["total_unique_records"],
        "report": "reports/flowfirst_data_coverage.json",
    }, indent=2))


if __name__ == "__main__":
    main()
