#!/usr/bin/env python3
"""SIT315 M2.T2C - render results.csv as speedup tables.

Run this on camera instead of showing the raw CSV.
    ./show_results.py
"""
import csv, sys

path = sys.argv[1] if len(sys.argv) > 1 else "results.csv"
rows = list(csv.DictReader(open(path)))

base, data = {}, {}
for r in rows:
    n, t, ms = int(r["elements"]), int(r["threads"]), float(r["median_ms"])
    if r["impl"] == "seq":
        base[n] = ms
    else:
        data[(r["impl"], n, t)] = ms

SIZES = sorted(base)
THREADS = [2, 4, 8, 16, 24, 32]
NAME = {10_000_000: "10M", 50_000_000: "50M", 100_000_000: "100M", 200_000_000: "200M"}

def table(impl, title):
    print(f"\n  {title}\n")
    head = f"  {'N':>5}  {'seq':>9}  " + "  ".join(f"{str(t)+'t':>15}" for t in THREADS)
    print(head)
    print("  " + "-" * (len(head) - 2))
    for n in SIZES:
        cells = []
        for t in THREADS:
            v = data.get((impl, n, t))
            cells.append(f"{v:>7,.0f} {base[n]/v:>6.2f}x" if v else f"{'-':>15}")
        print(f"  {NAME.get(n, n):>5}  {base[n]:>9,.0f}  " + "  ".join(cells))

table("omp", "OpenMP tasks, time in ms and speedup against sequential")
table("thread", "std::thread, time in ms and speedup against sequential")

print("\n  How far ahead OpenMP is at each setting\n")
head = f"  {'N':>5}  " + "  ".join(f"{str(t)+'t':>7}" for t in THREADS)
print(head)
print("  " + "-" * (len(head) - 2))
for n in SIZES:
    cells = []
    for t in THREADS:
        a, b = data.get(("omp", n, t)), data.get(("thread", n, t))
        cells.append(f"{b/a:>6.2f}x" if a and b else f"{'-':>7}")
    print(f"  {NAME.get(n, n):>5}  " + "  ".join(cells))

best = max(((base[n]/v, i, n, t) for (i, n, t), v in data.items()), key=lambda x: x[0])
print(f"\n  Best: {best[0]:.2f}x  ({best[1]}, {NAME.get(best[2])}, {best[3]} threads)\n")