#!/bin/bash
# SIT315 M4.T1D - timing sweep.
#
# Runs every citysim experiment in the report on this machine and writes
# results.csv. Every row carries the run's checksums, its exit code, its peak
# resident memory and the exact command that produced it.
#
#   source cuda_env.sh && ./run_experiments.sh
#   EXPERIMENTS="scaling" SIZES="M" REPEATS=1 ./run_experiments.sh
#
# Experiments, each selectable through EXPERIMENTS:
#   scaling   seq, and static, balanced and rayon at each thread count, on every city size
#   balance   the partition policies through the morning peak, under rush-hour, event and uniform demand
#   gpu       the cuda backend on every size, a block-size sweep, and the sensor pipeline on and off
#   stress    the 2896x2896 city on the CPU under a 12 GiB limit and on the GPU, then 4096x4096 on the GPU
#
# Scaling and gpu runs start from warm snapshots at 07:30, written once into DATA.

set -u

EXPERIMENTS=${EXPERIMENTS:-"scaling balance gpu stress"}
SIZES=${SIZES:-"S M L XL"}
REPEATS=${REPEATS:-5}
BIG_REPEATS=${BIG_REPEATS:-3}
BALANCE_THREADS=${BALANCE_THREADS:-"8 16 24 32"}
BLOCKS=${BLOCKS:-"32 64 128 256 512"}
OUT=${OUT:-results.csv}
DATA=${DATA:-data}
CUDA=${CUDA:-1}

declare -A CITY=([S]=32x32 [M]=128x128 [L]=512x512 [XL]=1024x1024)
declare -A SPAN=([S]=30m [M]=30m [L]=10m [XL]=5m)
declare -A THREADS=(
    [S]="1 2 4 6 8 10 12 14 16 20 24 28 32"
    [M]="1 2 4 6 8 10 12 14 16 20 24 28 32"
    [L]="1 2 4 8 12 16 24 32"
    [XL]="1 4 8 16 32"
)

cd "$(dirname "$0")"
mkdir -p "$DATA"

FEATURES=""
[ "$CUDA" = 1 ] && FEATURES="--features cuda"
cargo build --release --quiet $FEATURES || exit 1
BIN=$DATA/citysim-bench
cp target/release/citysim "$BIN" || exit 1
if [ "$CUDA" = 1 ]; then
    "$BIN" selftest > /dev/null || { echo "the GPU self-test failed; source cuda_env.sh or set CUDA=0" >&2; exit 1; }
fi

# The backend that writes snapshots: the GPU when there is one.
SNAPSHOT_BACKEND=static
[ "$CUDA" = 1 ] && SNAPSHOT_BACKEND=cuda

# Writes a snapshot of city SIZE at HH:MM, simulated from 06:00 with the extra
# flags given, once, and prints its path.
snapshot() {
    local size=$1 at=$2 name=$3
    shift 3
    local f="$DATA/${size}_${name}_${at/:/}.state"
    if [ ! -f "$f" ]; then
        local minutes=$(( (10#${at%%:*} - 6) * 60 + 10#${at##*:} ))
        "$BIN" run --city "${CITY[$size]}" --start 06:00 --duration "${minutes}m" --backend "$SNAPSHOT_BACKEND" \
            --save-state "$f" "$@" > /dev/null 2>&1 || { echo "cannot write $f" >&2; exit 1; }
    fi
    echo "$f"
}

# Reads a file once so the timed runs measure the program, not a cold disk.
warm() {
    cat "$1" > /dev/null
}

echo "experiment,scenario,run,backend,threads,block,cost,rebalance,city,link_cells,ticks,spawned,exited,parked,active,wall_ms,ms_per_tick,real_time_ratio,phase_a_ms,phase_b_ms,sync_ms,imbalance,window_imbalance,migration,repartitions,compile_ms,write_ms,state,sensor,exit,max_rss_kb,command" > "$OUT"

# citysim prints
# CSV,citysim,backend,threads,block,cost,rebalance,RxC,link_cells,ticks,spawned,exited,parked,active,wall_ms,
#   ms_per_tick,rtr,phase_a_ms,phase_b_ms,sync_ms,imbalance,window_imbalance,migration,repartitions,compile_ms,
#   write_ms,state,sensor
sim() {
    local experiment=$1 scenario=$2 run=$3 cmd=$4
    local line status rss backend threads ms_tick imb wimb
    /usr/bin/time -f %M -o "$DATA/.rss" bash -c "$cmd --csv" > "$DATA/.out" 2> /dev/null
    status=$?
    line=$(grep '^CSV,citysim,' "$DATA/.out")
    rss=$(tail -n 1 "$DATA/.rss" 2> /dev/null)
    if [ -z "$line" ]; then
        printf "  %-9s %-28s exit %s\n" "$scenario" "failed as expected?" "$status"
        echo "$experiment,$scenario,$run,,,,,,,,,,,,,,,,,,,,,,,,,,,$status,$rss,\"$cmd\"" >> "$OUT"
        return
    fi
    echo "$experiment,$scenario,$run,${line#CSV,citysim,},$status,$rss,\"$cmd\"" >> "$OUT"
    IFS=, read -r _ _ backend threads _ _ _ _ _ _ _ _ _ _ _ ms_tick _ _ _ _ imb wimb _ <<< "$line"
    printf "  %-9s %-9s %-2s threads  %9s ms/tick  imbalance %s %s\n" "$scenario" "$backend" "$threads" "$ms_tick" "$imb" "$wimb"
}

# The cut policies the balance experiment compares, as backend flags.
POLICIES=(
    "--backend static"
    "--backend balanced --cost cars --rebalance-every 1"
    "--backend balanced --cost cars --rebalance-every 0"
    "--backend balanced --cost time --rebalance-every 1"
    "--backend balanced --cost time --rebalance-every 12"
    "--backend rayon"
)

# A short name for a policy, for window-log file names.
policy_name() {
    echo "$1" | sed -e 's/--backend //' -e 's/ --cost /-/' -e 's/ --rebalance-every /-every/'
}

for experiment in $EXPERIMENTS; do
    case $experiment in
    scaling)
        for size in $SIZES; do
            snap=$(snapshot "$size" 07:30 rush)
            repeats=$REPEATS
            case $size in L | XL) repeats=$BIG_REPEATS ;; esac
            base="$BIN run --city ${CITY[$size]} --load-state $snap --duration ${SPAN[$size]}"
            echo "=== scaling, $size (${CITY[$size]}) ==="
            warm "$snap"
            for r in $(seq 1 "$repeats"); do
                sim scaling "$size" "$r" "$base --backend seq"
            done
            for backend in "static" "balanced --cost cars" "rayon"; do
                for t in ${THREADS[$size]}; do
                    for r in $(seq 1 "$repeats"); do
                        sim scaling "$size" "$r" "$base --backend $backend --threads $t"
                    done
                done
            done
        done
        ;;
    balance)
        for size in M L; do
            rows=${CITY[$size]%x*}
            cols=${CITY[$size]#*x}
            event="--event $((rows * 3 / 4)),$((cols * 3 / 4)),08:00,60"
            uniform="--peak 0 --floor 0.004 --edge 0"
            threads=$BALANCE_THREADS
            span=2h
            repeats=$BIG_REPEATS
            if [ "$size" = L ]; then
                threads="16 32"
                span=1h
                repeats=1
            fi
            for scenario in rush event uniform; do
                case $scenario in
                rush) snap=$(snapshot "$size" 07:00 rush); extra="" ;;
                event) snap=$(snapshot "$size" 07:00 rush); extra=$event ;;
                uniform) snap=$(snapshot "$size" 07:00 uniform $uniform); extra=$uniform ;;
                esac
                echo "=== balance, $size, $scenario ==="
                warm "$snap"
                for t in $threads; do
                    for policy in "${POLICIES[@]}"; do
                        log=$DATA/windows_${size}_${scenario}_$(policy_name "$policy")_$t.log
                        for r in $(seq 1 "$repeats"); do
                            sim balance "${size}_$scenario" "$r" \
                                "$BIN run --city ${CITY[$size]} --load-state $snap --duration $span $extra $policy --threads $t --window-log $log"
                        done
                    done
                done
            done
        done
        ;;
    gpu)
        if [ "$CUDA" != 1 ]; then
            echo "=== gpu skipped: CUDA=0 ==="
            continue
        fi
        for size in $SIZES; do
            snap=$(snapshot "$size" 07:30 rush)
            echo "=== gpu, $size (${CITY[$size]}) ==="
            warm "$snap"
            for r in $(seq 1 "$REPEATS"); do
                sim gpu "$size" "$r" "$BIN run --city ${CITY[$size]} --load-state $snap --duration ${SPAN[$size]} --backend cuda"
            done
        done
        snap=$(snapshot L 07:30 rush)
        echo "=== gpu, block size on L ==="
        for block in $BLOCKS; do
            for r in $(seq 1 "$REPEATS"); do
                sim gpu_block "L" "$r" "$BIN run --city ${CITY[L]} --load-state $snap --duration 10m --backend cuda --block $block"
            done
        done
        for size in M L; do
            snap=$(snapshot "$size" 07:30 rush)
            echo "=== gpu, sensor pipeline on $size ==="
            for backend in "cuda" "static --threads 16"; do
                for pipeline in on off; do
                    for r in $(seq 1 "$BIG_REPEATS"); do
                        sim pipeline "${size}_$pipeline" "$r" \
                            "$BIN run --city ${CITY[$size]} --load-state $snap --duration 1h --backend $backend --sensors $DATA/sensors_$size.csv --pipeline $pipeline"
                    done
                done
            done
        done
        ;;
    stress)
        echo "=== stress, 2896x2896 (16.9 GiB of car slots and metadata) ==="
        sim stress cpu_12GiB 1 "ulimit -v 12582912; exec $BIN run --city 2896x2896 --start 07:30 --duration 10m --backend static --threads 16"
        if [ "$CUDA" = 1 ]; then
            sim stress gpu 1 "$BIN run --city 2896x2896 --start 07:30 --duration 10m --backend cuda"
            echo "=== stress, 4096x4096 (34 GiB) ==="
            sim stress gpu_too_big 1 "$BIN run --city 4096x4096 --start 07:30 --duration 5m --backend cuda"
        fi
        ;;
    *)
        echo "unknown experiment: $experiment" >&2
        exit 1
        ;;
    esac
done
rm -f "$DATA/.rss" "$DATA/.out"
