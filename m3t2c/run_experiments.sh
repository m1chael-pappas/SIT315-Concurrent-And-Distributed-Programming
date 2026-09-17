#!/bin/bash
# SIT315 M3.T2C - timing sweep.
#
# Runs the two programs in this folder plus the three m2t2c baselines, so every
# number in the report comes from one script on one machine. Also sweeps the
# OpenCL local work size, which the report discusses separately.
#
#   ./run_experiments.sh
#   SIZES="1000000 4000000" PROCS="1 4" RUNS=1 ./run_experiments.sh
#
# Writes results.csv. Every row carries the exact command that produced it.

set -u

SIZES=${SIZES:-"1000000 4000000 16000000 64000000"}
PROCS=${PROCS:-"1 2 4 8 16"}
THREADS=${THREADS:-"8 16 32"}
RUNS=${RUNS:-3}
OUT=${OUT:-results.csv}

# Local work size sweep, run once at one size so the report can talk about it
# without the main sweep tripling in length. 0 means let the runtime choose.
LOCALS=${LOCALS:-"0 16 32 64 128 256 1024"}
LOCAL_N=${LOCAL_N:-16000000}
LOCAL_P=${LOCAL_P:-4}

# Anaconda ships the only working MPI on this machine, MPICH 4.3.2. Its mpic++
# wrapper points at a conda compiler that is not installed, so MPICH_CXX
# redirects it to the system g++. Both paths are absolute because a bare
# mpic++/mpirun resolves by PATH, and building with one MPI while launching with
# another leaves mpirun waiting on ranks that died on startup.
export MPICH_CXX=g++
MPI_BIN=${MPI_BIN:-$HOME/anaconda3/bin}
MPICXX=${MPICXX:-$MPI_BIN/mpic++}
MPIRUN=${MPIRUN:-$MPI_BIN/mpirun}
BASE=../m2t2c

for tool in "$MPICXX" "$MPIRUN"; do
    [ -x "$tool" ] || { echo "not executable: $tool  (set MPI_BIN)" >&2; exit 1; }
done

CXXFLAGS="-O2 -std=c++17 -Wall -Wextra"
$MPICXX $CXXFLAGS -o quicksort_mpi     quicksort_mpi.cpp                || exit 1
$MPICXX $CXXFLAGS -o quicksort_mpi_ocl quicksort_mpi_ocl.cpp -lOpenCL   || exit 1
g++ $CXXFLAGS           -o quicksort_sequential       $BASE/quicksort_sequential.cpp       || exit 1
g++ $CXXFLAGS -fopenmp  -o quicksort_parallel_openmp  $BASE/quicksort_parallel_openmp.cpp  || exit 1
g++ $CXXFLAGS -pthread  -o quicksort_parallel_threads $BASE/quicksort_parallel_threads.cpp || exit 1

echo "impl,elements,procs,threads,median_ms,scatter_ms,localsort_ms,pivots_ms,exchange_ms,merge_ms,gather_ms,checksum,device,local_ws,run,command" > "$OUT"

# The m2t2c programs print CSV,impl,elements,threads,pivot,median and carry the
# checksum on a separate line, so the phase columns stay empty for these rows.
baseline() {
    local impl=$1 cmd=$2 n=$3 t=$4 run=$5
    local out median checksum
    out=$(eval "$cmd" 2>/dev/null)
    median=$(echo "$out" | grep '^CSV,' | cut -d, -f6)
    checksum=$(echo "$out" | grep '^checksum' | awk '{print $2}')
    if [ -z "$median" ]; then
        echo "  FAILED: $cmd"
        echo "$impl,$n,1,$t,,,,,,,,,,,$run,\"$cmd\"" >> "$OUT"
        return
    fi
    echo "$impl,$n,1,$t,$median,,,,,,,$checksum,cpu native,,$run,\"$cmd\"" >> "$OUT"
    echo "  $impl n=$n threads=$t run=$run ${median}ms"
}

# The two programs here emit one CSV line with the phases already in order:
# impl,elements,procs,pivot,median,scatter,localsort,pivots,exchange,merge,gather,checksum[,device]
distributed() {
    local cmd=$1 run=$2 localws=$3
    local line impl n procs rest
    line=$(eval "$cmd" 2>/dev/null | grep '^CSV,')
    if [ -z "$line" ]; then
        echo "  FAILED: $cmd"
        echo ",,,,,,,,,,,,,$localws,$run,\"$cmd\"" >> "$OUT"
        return
    fi
    impl=$(echo "$line" | cut -d, -f2)
    n=$(echo "$line" | cut -d, -f3)
    procs=$(echo "$line" | cut -d, -f4)
    # skip the pivot/bitonic column, keep median through checksum, then device
    rest=$(echo "$line" | cut -d, -f6-)
    echo "$impl,$n,$procs,1,$rest,$localws,$run,\"$cmd\"" >> "$OUT"
    echo "  $impl n=$n procs=$procs local=$localws run=$run $(echo "$rest" | cut -d, -f1)ms"
}

for n in $SIZES; do
    echo "=== n=$n m2t2c baselines ==="
    for r in $(seq 1 $RUNS); do
        baseline seq "./quicksort_sequential --n $n --runs 1" "$n" 1 "$r"
    done
    for t in $THREADS; do
        for r in $(seq 1 $RUNS); do
            baseline omp     "./quicksort_parallel_openmp  --n $n --threads $t --runs 1" "$n" "$t" "$r"
            baseline threads "./quicksort_parallel_threads --n $n --threads $t --runs 1" "$n" "$t" "$r"
        done
    done

    echo "=== n=$n mpi ==="
    for p in $PROCS; do
        for r in $(seq 1 $RUNS); do
            distributed "$MPIRUN -np $p ./quicksort_mpi --n $n --runs 1" "$r" ""
        done
    done

    echo "=== n=$n mpi+ocl ==="
    for p in $PROCS; do
        for r in $(seq 1 $RUNS); do
            distributed "$MPIRUN -np $p ./quicksort_mpi_ocl --n $n --runs 1" "$r" "runtime"
        done
    done
done

echo "=== local work size sweep, n=$LOCAL_N, np=$LOCAL_P ==="
for ws in $LOCALS; do
    for r in $(seq 1 $RUNS); do
        distributed "$MPIRUN -np $LOCAL_P ./quicksort_mpi_ocl --n $LOCAL_N --local $ws --runs 1" "$r" "$ws"
    done
done

echo "done, $(($(wc -l < "$OUT") - 1)) rows in $OUT"
