#!/usr/bin/env bash
#
# SIT315 M2.T3D benchmark sweep.
#
#   ./run_benchmarks.sh                    full sweep, 3 runs per config
#   RUNS=5 ./run_benchmarks.sh             5 runs per config
#   ./run_benchmarks.sh small medium       named workloads only
#
# Writes results.csv, then renders it with show_results.py.

set -euo pipefail

BIN=./traffic_sim
RUNS=${RUNS:-3}
OUT=${OUT:-results.csv}
DATADIR=${DATADIR:-data}

if [ ! -x "$BIN" ]; then
    echo "building traffic_sim"
    g++ -std=c++17 -O2 -pthread -Wall -o traffic_sim traffic_sim.cpp
fi

mkdir -p "$DATADIR"

# name    lights hours          rows      approx size
# small     500    336     2,016,000            54 MiB
# medium   1000   1680    20,160,000           540 MiB
# large    1500   3360    60,480,000           1.6 GiB
declare -A LIGHTS=( [small]=500  [medium]=1000 [large]=1500 )
declare -A HOURS=(  [small]=336  [medium]=1680 [large]=3360 )

if [ $# -gt 0 ]; then
    WORKLOADS=("$@")
else
    WORKLOADS=(small medium large)
fi

# producers consumers buffer block summary
CONFIGS=(
    "1  1   64  4096 local"
    "2  2   64  4096 local"
    "4  4   64  4096 local"
    "8  8   64  4096 local"
    "16 16  64  4096 local"
    "8  2   64  4096 local"
    "2  8   64  4096 local"
    "8  8    4  4096 local"
    "8  8  256  4096 local"
    "8  8   64   256 local"
    "8  8   64  4096 atomic"
    "8  8   64  4096 shared"
)

echo "workload,rows,mode,producers,consumers,buffer,block,summary,run,ms,checksum,records" > "$OUT"

emit()
{
    local workload=$1 rows=$2 run=$3 line=$4
    local tag mode p c buf blk summary recs ms checksum
    IFS=, read -r tag mode p c buf blk summary recs ms checksum <<< "$line"
    echo "$workload,$rows,$mode,$p,$c,$buf,$blk,$summary,$run,$ms,$checksum,$recs" >> "$OUT"
}

for w in "${WORKLOADS[@]}"; do
    f="$DATADIR/traffic_$w.csv"
    if [ ! -f "$f" ]; then
        echo "generating $f"
        $BIN gen --out "$f" --lights "${LIGHTS[$w]}" --hours "${HOURS[$w]}" | sed -n '2,6p'
    fi
    rows=$(( LIGHTS[$w] * 12 * HOURS[$w] ))

    # Warm the page cache first. Without this the first timed run pays for the
    # cold read off disk and the comparison measures storage, not threading.
    $BIN seq --in "$f" --print-hours 0 > /dev/null

    echo "  $w  sequential baseline"
    for r in $(seq 1 "$RUNS"); do
        emit "$w" "$rows" "$r" "$($BIN seq --in "$f" --print-hours 0 --csv | grep '^CSV,')"
    done

    for cfg in "${CONFIGS[@]}"; do
        read -r p c b blk s <<< "$cfg"
        echo "  $w  P=$p C=$c buffer=$b block=$blk summary=$s"
        for r in $(seq 1 "$RUNS"); do
            emit "$w" "$rows" "$r" "$($BIN par --in "$f" --producers "$p" --consumers "$c" \
                --buffer "$b" --block "$blk" --summary "$s" --print-hours 0 --csv | grep '^CSV,')"
        done
    done
done

echo
echo "wrote $OUT"
python3 show_results.py "$OUT"