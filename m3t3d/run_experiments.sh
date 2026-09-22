#!/bin/bash
# SIT315 M3.T3D - timing sweep.
#
# Runs traffic_mr and the m2t3d baselines, so every number in the report comes
# from one script on one machine.
#
#   ./run_experiments.sh
#   EXPERIMENTS="scaling" WORKLOADS="small" PROCS="1 4" REPEATS=1 ./run_experiments.sh
#
# Experiments, each selectable through EXPERIMENTS:
#   scaling   uniform workloads at every process count, plus m2t3d seq and par
#   skew      the five partitions on the two skewed workloads
#   combiner  combiner on against off, which is the message cost of the shuffle
#   reducers  fewer reducers than mappers at a fixed process count
#   order     the same workload sorted and shuffled, which is what locality buys
#
# Writes results.csv. Every row carries the exact command that produced it.

set -u

EXPERIMENTS=${EXPERIMENTS:-"scaling skew combiner reducers order"}
WORKLOADS=${WORKLOADS:-"small medium large xlarge"}
PROCS=${PROCS:-"1 2 4 8 16"}
SKEW_PROCS=${SKEW_PROCS:-"1 2 4 8 16"}
PARTITIONS=${PARTITIONS:-"block hash weighted light hybrid"}
REDUCERS=${REDUCERS:-"1 2 4 8 16"}
REPEATS=${REPEATS:-3}
OUT=${OUT:-results.csv}
DATA=${DATA:-data}
M2=../m2t3d

# Anaconda ships the only working MPI on this machine, MPICH 4.3.2. Its mpic++
# wrapper points at a conda compiler that is not installed, so MPICH_CXX
# redirects it to the system g++. Both paths are absolute because a bare
# mpic++/mpirun resolves by PATH, and building with one MPI while launching with
# another leaves mpirun waiting on ranks that died on startup.
export MPICH_CXX=g++
MPI_BIN=${MPI_BIN:-$HOME/anaconda3/bin}
MPICXX=${MPICXX:-$MPI_BIN/mpic++}
MPIRUN=${MPIRUN:-$MPI_BIN/mpirun}

for tool in "$MPICXX" "$MPIRUN"; do
    [ -x "$tool" ] || { echo "not executable: $tool  (set MPI_BIN)" >&2; exit 1; }
done

cd "$(dirname "$0")"
mkdir -p "$DATA"

$MPICXX -std=c++17 -O2 -Wall -Wextra -o traffic_mr traffic_mr.cpp                   || exit 1
g++     -std=c++17 -O2 -Wall -Wextra -o traffic_gen_skew traffic_gen_skew.cpp       || exit 1
g++     -std=c++17 -O2 -pthread -Wall -o traffic_sim $M2/traffic_sim.cpp            || exit 1

declare -A LIGHTS=( [small]=500 [medium]=1000 [large]=1500 [xlarge]=2000 )
declare -A HOURS=(  [small]=336 [medium]=1680 [large]=3360 [xlarge]=5040 )

# Uniform workloads are the m2t3d files, so the checksums match that task's results.
uniform() {
    local w=$1
    local f="$M2/data/traffic_$w.csv"
    if [ ! -f "$f" ]; then
        mkdir -p "$M2/data"
        ./traffic_sim gen --out "$f" --lights "${LIGHTS[$w]}" --hours "${HOURS[$w]}" > /dev/null || exit 1
    fi
    echo "$f"
}

skewed() {
    local name=$1 share=$2
    local f="$DATA/$name.csv"
    if [ ! -f "$f" ]; then
        ./traffic_gen_skew --out "$f" --lights 1500 --hours 3360 --incident-share "$share" > /dev/null || exit 1
    fi
    echo "$f"
}

# A reproducible stream of pseudo random bytes from a seed, the recipe in the
# coreutils manual under "Sources of random data" (info coreutils 'Random sources').
seeded_random() {
    openssl enc -aes-256-ctr -pass pass:"$1" -nosalt < /dev/zero 2> /dev/null
}

# The same rows as the medium workload in a fixed pseudo random order.
shuffled() {
    local f="$DATA/traffic_medium_shuffled.csv"
    if [ ! -f "$f" ]; then
        shuf --random-source=<(seeded_random 315) "$(uniform medium)" > "$f" || exit 1
    fi
    echo "$f"
}

# Reads a file once so the timed runs measure the program, not a cold disk.
warm() {
    cat "$1" > /dev/null
}

echo "experiment,workload,rows,impl,procs,reducers,partition,combiner,keys,shuffle_entries,remote_entries,entry_imbalance,time_imbalance,map_ms,plan_ms,shuffle_ms,reduce_ms,gather_ms,total_ms,checksum,run,command" > "$OUT"

# traffic_mr prints
# CSV,mr,procs,reducers,partition,combiner,records,keys,mapped,remote,entry_imb,time_imb,map,plan,shuffle,reduce,gather,total,checksum
mr() {
    local experiment=$1 workload=$2 run=$3 cmd=$4
    local line procs reducers part comb records keys mapped remote eimb timb
    local map plan shuf red gath total checksum
    line=$(eval "$cmd" 2>/dev/null | grep '^CSV,mr,')
    if [ -z "$line" ]; then
        echo "  FAILED: $cmd"
        echo "$experiment,$workload,,mr,,,,,,,,,,,,,,,,,$run,\"$cmd\"" >> "$OUT"
        return
    fi
    IFS=, read -r _ _ procs reducers part comb records keys mapped remote eimb timb \
        map plan shuf red gath total checksum <<< "$line"
    echo "$experiment,$workload,$records,mr,$procs,$reducers,$part,$comb,$keys,$mapped,$remote,$eimb,$timb,$map,$plan,$shuf,$red,$gath,$total,$checksum,$run,\"$cmd\"" >> "$OUT"
    printf "  %-9s np=%-2s R=%-2s %-8s combine=%-3s imbalance=%-6s total=%9s ms\n" \
        "$workload" "$procs" "$reducers" "$part" "$comb" "$eimb" "$total"
}

# m2t3d prints CSV,seq,1,0,0,0,none,records,ms,checksum and
# CSV,par,producers,consumers,buffer,block,summary,records,ms,checksum.
# Producers land in the procs column and consumers in the reducers column,
# which is the role each plays.
m2() {
    local experiment=$1 workload=$2 run=$3 cmd=$4
    local line mode producers consumers summary records ms checksum
    line=$(eval "$cmd" 2>/dev/null | grep '^CSV,')
    if [ -z "$line" ]; then
        echo "  FAILED: $cmd"
        echo "$experiment,$workload,,m2t3d,,,,,,,,,,,,,,,,,$run,\"$cmd\"" >> "$OUT"
        return
    fi
    IFS=, read -r _ mode producers consumers _ _ summary records ms checksum <<< "$line"
    echo "$experiment,$workload,$records,m2t3d-$mode,$producers,$consumers,$summary,,,,,,,,,,,,$ms,$checksum,$run,\"$cmd\"" >> "$OUT"
    printf "  %-9s m2t3d %-3s P=%-2s C=%-2s total=%9s ms\n" "$workload" "$mode" "$producers" "$consumers" "$ms"
}

MRFLAGS="--print-hours 0 --verify --csv"

for experiment in $EXPERIMENTS; do
    case $experiment in
    scaling)
        for w in $WORKLOADS; do
            f=$(uniform "$w")
            echo "=== scaling, $w ==="
            warm "$f"
            for r in $(seq 1 "$REPEATS"); do
                m2 scaling "$w" "$r" "./traffic_sim seq --in $f --print-hours 0 --csv"
                m2 scaling "$w" "$r" "./traffic_sim par --in $f --producers 16 --consumers 16 --print-hours 0 --csv"
            done
            for p in $PROCS; do
                for r in $(seq 1 "$REPEATS"); do
                    mr scaling "$w" "$r" "$MPIRUN -np $p ./traffic_mr --in $f --partition weighted $MRFLAGS"
                done
            done
        done
        ;;
    skew)
        for spec in skew_rollout:0 skew_incident:0.25; do
            name=${spec%%:*}
            f=$(skewed "$name" "${spec##*:}")
            echo "=== skew, $name ==="
            warm "$f"
            for part in $PARTITIONS; do
                for p in $SKEW_PROCS; do
                    for r in $(seq 1 "$REPEATS"); do
                        mr skew "$name" "$r" "$MPIRUN -np $p ./traffic_mr --in $f --partition $part $MRFLAGS"
                    done
                done
            done
        done
        ;;
    combiner)
        f=$(uniform medium)
        echo "=== combiner, medium ==="
        warm "$f"
        for p in 1 4 16; do
            for flag in "" "--no-combine"; do
                for r in $(seq 1 "$REPEATS"); do
                    mr combiner medium "$r" "$MPIRUN -np $p ./traffic_mr --in $f --partition weighted $flag $MRFLAGS"
                done
            done
        done
        ;;
    reducers)
        f=$(uniform large)
        echo "=== reducers, large, 16 processes ==="
        warm "$f"
        for red in $REDUCERS; do
            for r in $(seq 1 "$REPEATS"); do
                mr reducers large "$r" "$MPIRUN -np 16 ./traffic_mr --in $f --partition weighted --reducers $red $MRFLAGS"
            done
        done
        ;;
    order)
        for f in "$(uniform medium)" "$(shuffled)"; do
            case $f in *shuffled*) name=medium_shuffled ;; *) name=medium ;; esac
            echo "=== order, $name ==="
            warm "$f"
            for part in $PARTITIONS; do
                for r in $(seq 1 "$REPEATS"); do
                    mr order "$name" "$r" "$MPIRUN -np 16 ./traffic_mr --in $f --partition $part $MRFLAGS"
                done
            done
        done
        ;;
    *)
        echo "unknown experiment: $experiment" >&2
        exit 1
        ;;
    esac
done

echo "done, $(($(wc -l < "$OUT") - 1)) rows in $OUT"
