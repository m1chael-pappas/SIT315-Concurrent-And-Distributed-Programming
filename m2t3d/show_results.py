#!/usr/bin/env python3
"""Render results.csv from run_benchmarks.sh as speedup tables.

    python3 show_results.py [results.csv]
"""

import csv
import statistics
import sys
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "results.csv"

rows = []
with open(path, newline="") as handle:
    for row in csv.DictReader(handle):
        row["ms"] = float(row["ms"])
        row["rows"] = int(row["rows"])
        rows.append(row)

if not rows:
    sys.exit("no rows in " + path)

# group repeated runs of the same configuration
grouped = defaultdict(list)
for row in rows:
    key = (row["workload"], row["rows"], row["mode"], row["producers"],
           row["consumers"], row["buffer"], row["block"], row["summary"])
    grouped[key].append(row["ms"])

checksums = defaultdict(set)
for row in rows:
    checksums[row["workload"]].add(row["checksum"])

baseline = {}
for key, times in grouped.items():
    if key[2] == "seq":
        baseline[key[0]] = statistics.median(times)

order = {}
for i, row in enumerate(rows):
    key = (row["workload"], row["rows"], row["mode"], row["producers"],
           row["consumers"], row["buffer"], row["block"], row["summary"])
    order.setdefault(key, i)

workloads = []
for row in rows:
    if row["workload"] not in workloads:
        workloads.append(row["workload"])

for workload in workloads:
    keys = sorted([k for k in grouped if k[0] == workload], key=lambda k: order[k])
    total_rows = keys[0][1]
    base = baseline.get(workload)

    print()
    print("=" * 78)
    print(f"{workload}  ({total_rows:,} rows)")
    unique = checksums[workload]
    print(f"checksum: {'identical across every run (' + next(iter(unique)) + ')' if len(unique) == 1 else 'MISMATCH ' + str(unique)}")
    print("=" * 78)
    print(f"{'config':<34}{'median ms':>12}{'min ms':>10}{'spread':>9}{'speedup':>10}")
    print("-" * 78)

    for key in keys:
        _, _, mode, p, c, buf, blk, summary = key
        times = grouped[key]
        med = statistics.median(times)
        lo = min(times)
        spread = (max(times) - lo) / lo * 100 if lo else 0
        if mode == "seq":
            label = "sequential baseline"
            speedup = "1.00x"
        else:
            label = f"P={p} C={c} buf={buf} block={blk} {summary}"
            speedup = f"{base / med:.2f}x" if base else "-"
        print(f"{label:<34}{med:>12.1f}{lo:>10.1f}{spread:>8.1f}%{speedup:>10}")

print()