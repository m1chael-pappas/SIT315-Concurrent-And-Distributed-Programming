#!/usr/bin/env bash
# SIT315 M2.T2C - pivot rule comparison, random and already-sorted input.
set -euo pipefail
cd "$(dirname "$0")"

N=${1:-50000000}
THREADS=${2:-16}
RUNS=3
CSV=pivot_results.csv
echo "impl,elements,threads,pivot,median_ms" > "$CSV"

for p in median3 middle last first; do
  echo "=== sequential, random input, pivot=$p ==="
  ./quicksort_sequential --n "$N" --pivot "$p" --runs "$RUNS" \
    | tee /dev/stderr | grep '^CSV,' | cut -d, -f2- >> "$CSV"
  echo
  echo "=== openmp $THREADS threads, random input, pivot=$p ==="
  ./quicksort_parallel_openmp --n "$N" --threads "$THREADS" --pivot "$p" --runs "$RUNS" \
    | tee /dev/stderr | grep '^CSV,' | cut -d, -f2- >> "$CSV"
  echo
done

echo "=== already-sorted input, 3M elements, 60s timeout each ==="
for p in median3 middle last; do
  printf "pivot=%-8s " "$p"
  timeout 60 ./quicksort_sequential --n 3000000 --pattern sorted --pivot "$p" --runs 1 \
    | grep '^run 1' || echo "DNF, killed at 60s"
done

column -s, -t < "$CSV"