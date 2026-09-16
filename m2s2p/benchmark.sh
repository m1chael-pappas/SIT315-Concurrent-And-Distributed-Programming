#!/bin/bash
#
# SIT315 - M2.S2P - Activity 2, steps 5 and 6
# Runs the thread count sweep and the partition size sweep, and prints the
# results as markdown tables ready to paste into the report.
#
# Usage: ./benchmark.sh [size] [repeats]

SIZE=${1:-100000000}
REPEATS=${2:-3}

g++ -O2 -std=c++17 -pthread -o VectorAddParallel VectorAddParallel.cpp || exit 1

echo "cores available: $(nproc)"
echo "size: $SIZE, repeats per configuration: $REPEATS (best time kept)"
echo ""

# pulls one field out of the program output, e.g. field add_us
field() {
  echo "$1" | tr ' ' '\n' | grep "^$2=" | cut -d= -f2
}

# runs one configuration REPEATS times and echoes "gen add total"
run_best() {
  local best_add=999999999
  local best_gen=999999999
  for r in $(seq 1 "$REPEATS"); do
    out=$(./VectorAddParallel "$SIZE" "$1" "$2" "$3")
    a=$(field "$out" add_us)
    g=$(field "$out" gen_us)
    [ "$a" -lt "$best_add" ] && best_add=$a
    [ "$g" -lt "$best_gen" ] && best_gen=$g
  done
  echo "$best_gen $best_add"
}

echo "## Thread count sweep (static block decomposition)"
echo ""
echo "| Threads | Generation (us) | Addition (us) | Total (us) | Speedup (addition) | Speedup (total) |"
echo "|---|---|---|---|---|---|"

BASE_ADD=0
BASE_TOTAL=0

for T in 1 2 4 8 16 32 64; do
  read -r g a <<< "$(run_best "$T" 0 0)"
  total=$((g + a))
  if [ "$T" -eq 1 ]; then
    BASE_ADD=$a
    BASE_TOTAL=$total
  fi
  sa=$(awk "BEGIN{printf \"%.2f\", $BASE_ADD/$a}")
  st=$(awk "BEGIN{printf \"%.2f\", $BASE_TOTAL/$total}")
  echo "| $T | $g | $a | $total | ${sa}x | ${st}x |"
done

echo ""
echo "## Partition size sweep (dynamic chunks, threads = $(nproc))"
echo ""
echo "| Partition size | Work units | Addition (us) |"
echo "|---|---|---|"

NT=$(nproc)
for P in 1024 8192 65536 524288 4194304 33554432; do
  read -r g a <<< "$(run_best "$NT" "$P" 1)"
  units=$(( (SIZE + P - 1) / P ))
  echo "| $P | $units | $a |"
done

echo ""
echo "## Static vs dynamic at $(nproc) threads"
echo ""
NT=$(nproc)
read -r g a <<< "$(run_best "$NT" 0 0)"
echo "static block, $NT threads: add=$a us"
read -r g a <<< "$(run_best "$NT" 524288 1)"
echo "dynamic 524288,  $NT threads: add=$a us"