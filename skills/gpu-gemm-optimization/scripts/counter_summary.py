#!/usr/bin/env python3
"""Summarize rocprofv3 counter CSV without silently mixing dispatch dimensions."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

REQUIRED = {
    "Dispatch_Id", "Agent_Id", "Kernel_Name", "Grid_Size",
    "Workgroup_Size", "Counter_Name", "Counter_Value",
}


def summarize_rows(rows, sum_dimensions=False):
    # Process/Queue disambiguate dispatch ids when a CSV contains multiple processes.
    dispatches = defaultdict(list)
    for row in rows:
        missing = REQUIRED - row.keys()
        if missing:
            raise ValueError(f"missing rocprofv3 columns: {sorted(missing)}")
        config = tuple(row[k] for k in ("Agent_Id", "Kernel_Name", "Grid_Size", "Workgroup_Size"))
        key = (config, row.get("Process_Id", ""), row.get("Queue_Id", ""),
               row["Dispatch_Id"], row["Counter_Name"])
        value = float(row["Counter_Value"])
        if not math.isfinite(value):
            raise ValueError(f"non-finite counter {key}: {value}")
        dispatches[key].append(value)
    configs = defaultdict(lambda: defaultdict(list))
    for (config, process, queue, dispatch, counter), values in dispatches.items():
        if len(values) > 1 and not sum_dimensions:
            raise ValueError(
                f"multiple rows for dispatch {dispatch}/{process}/{queue}, counter {counter}. "
                "Check dimensions/duplicates. Only use --sum-dimensions for additive raw counts."
            )
        configs[config][counter].append(sum(values))
    output = []
    for config, counters in sorted(configs.items()):
        output.append({
            **dict(zip(("agent", "kernel", "grid_size", "workgroup_size"), config)),
            "counters": {
                name: {"dispatch_count": len(values), "mean_per_dispatch": statistics.mean(values),
                       "median_per_dispatch": statistics.median(values),
                       "min": min(values), "max": max(values)}
                for name, values in sorted(counters.items())
            },
        })
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="+", type=Path)
    parser.add_argument("--sum-dimensions", action="store_true",
                        help="Explicitly sum duplicate dispatch/counter rows; NOT valid for all metrics.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results = []
    for path in args.csv:
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            missing = REQUIRED - set(reader.fieldnames or [])
            if missing:
                parser.error(f"{path}: missing columns {sorted(missing)}")
            try:
                groups = summarize_rows(list(reader), args.sum_dimensions)
            except ValueError as exc:
                parser.error(f"{path}: {exc}")
        if not groups:
            parser.error(f"{path}: no dispatch counters; check kernel filter/iteration range")
        results.append({
            "path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "dimension_policy": "explicit-sum" if args.sum_dimensions else "one-row-per-dispatch-counter",
            "warning": "Units/dimensions must be obtained from rocprofv3-avail on the measured GPU. "
                       "Wait counts are not wall-clock latency. Files/passes are not merged.",
            "groups": groups,
        })
    text = json.dumps(results, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
