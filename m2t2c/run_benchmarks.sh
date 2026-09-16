#!/usr/bin/env bash
# SIT315 M2.T2C - workload and thread sweep. Writes results.csv + bench_log.txt.
set -euo pipefail
cd "$(dirname "$0")"

SIZES=(10000000 50000000 100000000 200000000)
THREADS=(2 4 8 16 24 32)
RUNS=5
PIVOT=median3
SEED=315

LOG=bench_log.txt
CSV=results.csv
: > "$LOG"
echo "impl,elements,threads,pivot,median_ms" > "$CSV"

run() {
  echo "=== $* ===" | tee -a "$LOG"
  "$@" | tee -a "$LOG" | grep '^CSV,' | cut -d, -f2- >> "$CSV"
  echo | tee -a "$LOG"
}

for n in "${SIZES[@]}"; do
  run ./quicksort_sequential --n "$n" --seed "$SEED" --pivot "$PIVOT" --runs "$RUNS"
done

for n in "${SIZES[@]}"; do
  for t in "${THREADS[@]}"; do
    run ./quicksort_parallel_openmp  --n "$n" --seed "$SEED" --threads "$t" \
        --pivot "$PIVOT" --runs "$RUNS" --task-cutoff 100000
    run ./quicksort_parallel_threads --n "$n" --seed "$SEED" --threads "$t" \
        --pivot "$PIVOT" --runs "$RUNS" --size-cutoff 100000
  done
done

echo "done: $CSV"
column -s, -t < "$CSV"