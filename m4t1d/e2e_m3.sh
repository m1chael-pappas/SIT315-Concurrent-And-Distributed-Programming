#!/bin/bash
# SIT315 M4.T1D - end-to-end check against m2t3d and m3t3d.
#
# Simulates a city, writes its sensor CSV, and hands that file to the m2t3d
# threaded analyzer and the m3t3d MPI MapReduce analyzer. The check passes when
# all three programs print the same sensor checksum and the two analyzers write
# byte-identical rankings.
#
#   ./e2e_m3.sh
#   CITY=64x64 DURATION=6h PROCS=8 ./e2e_m3.sh

set -u

CITY=${CITY:-16x16}
START=${START:-06:00}
DURATION=${DURATION:-4h}
PEAK=${PEAK:-08:00}
PROCS=${PROCS:-4}
TOP=${TOP:-5}
DATA=${DATA:-data}
CUDA=${CUDA:-1}
M2=../m2t3d
M3=../m3t3d

# The same MPI setup as m3t3d/run_experiments.sh: Anaconda's MPICH, with its
# mpic++ wrapper pointed at the system g++.
export MPICH_CXX=g++
MPI_BIN=${MPI_BIN:-$HOME/anaconda3/bin}
MPICXX=${MPICXX:-$MPI_BIN/mpic++}
MPIRUN=${MPIRUN:-$MPI_BIN/mpirun}

for tool in "$MPICXX" "$MPIRUN"; do
    [ -x "$tool" ] || { echo "not executable: $tool  (set MPI_BIN)" >&2; exit 1; }
done

cd "$(dirname "$0")"
mkdir -p "$DATA"

# The same features as run_experiments.sh, so the check never swaps a GPU build
# for a CPU-only one. The cuda feature builds without a GPU.
FEATURES=""
[ "$CUDA" = 1 ] && FEATURES="--features cuda"
cargo build --release --quiet $FEATURES                                             || exit 1
$MPICXX -std=c++17 -O2 -Wall -Wextra -o "$DATA/traffic_mr" $M3/traffic_mr.cpp       || exit 1
g++     -std=c++17 -O2 -pthread -Wall -o "$DATA/traffic_sim" $M2/traffic_sim.cpp    || exit 1

csv=$DATA/e2e_$CITY.csv
m2_out=$DATA/e2e_$CITY.m2.out
m3_out=$DATA/e2e_$CITY.m3.out
log=$DATA/e2e_$CITY.log
: > "$log"

echo "== citysim run --city $CITY --start $START --duration $DURATION --sensors $csv"
sim=$(./target/release/citysim run --city "$CITY" --start "$START" --duration "$DURATION" --sensors "$csv" 2>>"$log") \
    || { echo "citysim failed, see $log" >&2; exit 1; }
sim_sum=$(awk '/^checksums/ {print $NF}' <<<"$sim")

echo "== traffic_sim seq --in $csv --top $TOP --out-summary $m2_out"
m2_sum=$("$DATA/traffic_sim" seq --in "$csv" --top "$TOP" --print-hours 0 --out-summary "$m2_out" \
    2>>"$log" | awk '/checksum/ {print $NF}')

echo "== mpirun -np $PROCS traffic_mr --in $csv --top $TOP --verify --out-summary $m3_out"
m3_sum=$("$MPIRUN" -np "$PROCS" "$DATA/traffic_mr" --in "$csv" --top "$TOP" --print-hours 0 --verify \
    --out-summary "$m3_out" 2>>"$log" | awk '/^checksum/ {print $NF}')

echo
printf '%-10s %s\n' citysim "$sim_sum" m2t3d "$m2_sum" m3t3d "$m3_sum"

# Ranked lights of the peak hour, with each id turned back into its grid position.
cols=${CITY#*x}
awk -F, -v hour="$PEAK" -v cols="$cols" '
    substr($1, 12) == hour {
        printf "\n%s ranking (row, col):", hour
        for (i = 2; i <= NF; i++) {
            split($i, kv, ":")
            printf "  TL%04d (%d, %d) %d", kv[1], int(kv[1] / cols), kv[1] % cols, kv[2]
        }
        print ""
    }' "$m3_out"

echo
if [ -n "$sim_sum" ] && [ "$sim_sum" = "$m2_sum" ] && [ "$sim_sum" = "$m3_sum" ] && cmp -s "$m2_out" "$m3_out"; then
    echo "PASS  checksums match and the m2t3d and m3t3d rankings are identical"
else
    echo "FAIL  checksums or rankings differ, see $log"
    exit 1
fi
