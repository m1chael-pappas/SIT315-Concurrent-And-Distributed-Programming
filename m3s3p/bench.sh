#!/usr/bin/env bash
# Builds both vector addition programs and runs them across a few sizes.
# Prints a Markdown table you can paste into the report.
set -e
cd "$(dirname "$0")"
g++ -O2 vector_ops.cpp -o vector_ops -lOpenCL
g++ -O2 -pthread vector_add_threads.cpp -o vector_add_threads
THREADS=${THREADS:-$(nproc)}
RUNS=${RUNS:-7}

median() { sort -n | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}'; }

echo "| Size | Sequential (us) | std::thread x$THREADS (us) | OpenCL kernel (us) | OpenCL copy+kernel+read (us) |"
echo "|---|---|---|---|---|"
for n in 100000 1000000 10000000 50000000; do
  seq=(); thr=(); ker=(); ocl=()
  for r in $(seq $RUNS); do
    line=$(./vector_add_threads $n $THREADS)
    seq+=( $(echo "$line" | sed -E 's/.*sequential=([0-9]+).*/\1/') )
    thr+=( $(echo "$line" | sed -E 's/.*threaded=([0-9]+).*/\1/') )
    line=$(./vector_ops $n | tail -1)
    ker+=( $(echo "$line" | sed -E 's/.*kernel=([0-9]+) us copy.*/\1/') )
    ocl+=( $(echo "$line" | sed -E 's/.*read=([0-9]+).*/\1/') )
  done
  printf "| %s | %s | %s | %s | %s |\n" "$n" \
    "$(printf '%s\n' "${seq[@]}" | median)" \
    "$(printf '%s\n' "${thr[@]}" | median)" \
    "$(printf '%s\n' "${ker[@]}" | median)" \
    "$(printf '%s\n' "${ocl[@]}" | median)"
done
