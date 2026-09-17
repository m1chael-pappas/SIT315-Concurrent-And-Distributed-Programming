#!/bin/bash
# SIT315 M3.T1P - timing sweep for the three MPI matrix multiplication programs
# plus the m2t1p CPU baselines, so every number in the report comes from one run
# of this script on one machine.
#
#   ./run_experiments.sh
#   SIZES="500 1000" PROCS="1 4" REPEATS=1 ./run_experiments.sh
#
# Writes results.csv. Every row carries the exact command that produced it.

set -u

SIZES=${SIZES:-"100 500 1000 2000"}
PROCS=${PROCS:-"1 2 4 8 16"}
THREADS=${THREADS:-"1 2 4 8"}
REPEATS=${REPEATS:-3}
OUT=${OUT:-results.csv}

# Anaconda ships the only working MPI on this machine, MPICH 4.3.2. Its mpic++
# wrapper points at a conda compiler that is not installed, so MPICH_CXX
# redirects it to the system g++. The system OpenMPI and MPICH both hang in
# MPI_Init on this WSL2 setup.
#
# Both paths are absolute on purpose. A bare mpic++/mpirun resolves by PATH, and
# a shell without Anaconda on PATH picks the system OpenMPI wrappers instead.
# Building with one MPI and launching with the other leaves mpirun waiting on
# ranks that died on startup, which looks exactly like a hang.
export MPICH_CXX=g++
MPI_BIN=${MPI_BIN:-$HOME/anaconda3/bin}
MPICXX=${MPICXX:-$MPI_BIN/mpic++}
MPIRUN=${MPIRUN:-$MPI_BIN/mpirun}

for tool in "$MPICXX" "$MPIRUN"; do
    [ -x "$tool" ] || { echo "not executable: $tool  (set MPI_BIN)" >&2; exit 1; }
done
BASE=../m2t1p

$MPICXX -O2 -o matrix_mpi matrix_mpi.cpp || exit 1
$MPICXX -O2 -fopenmp -o matrix_mpi_omp matrix_mpi_omp.cpp || exit 1
$MPICXX -O2 -o matrix_mpi_ocl matrix_mpi_ocl.cpp -lOpenCL || exit 1
g++ -O2 -o matrix_seq $BASE/matrix_seq.cpp || exit 1
g++ -O2 -fopenmp -o matrix_omp $BASE/matrix_omp.cpp || exit 1

echo "version,N,procs,threads,generate_us,distribute_us,compute_us,gather_us,seconds,checksum,verify,run,command" > "$OUT"

# The m2t1p programs print one line and no phase breakdown, so the phase columns
# stay empty for these rows.
baseline() {
    local version=$1 cmd=$2 n=$3 t=$4 run=$5
    local line seconds checksum
    line=$(eval "$cmd" 2>/dev/null | grep '^version=')
    seconds=$(echo "$line" | sed -n 's/.*time=\([0-9.]*\) s.*/\1/p')
    checksum=$(echo "$line" | sed -n 's/.*checksum=\([0-9]*\).*/\1/p')
    echo "$version,$n,1,$t,,,,,$seconds,$checksum,checksum-only,$run,\"$cmd\"" >> "$OUT"
    echo "  $version N=$n threads=$t run=$run ${seconds}s"
}

# The three MPI programs all emit a CSV line with the phase breakdown already in
# the right order, so the row only needs the run number and command appended.
mpirun_row() {
    local cmd=$1 run=$2
    local line
    line=$(eval "$cmd" 2>/dev/null | grep '^CSV,' | cut -d, -f2-)
    if [ -z "$line" ]; then
        echo "  FAILED: $cmd"
        echo ",,,,,,,,,,failed,$run,\"$cmd\"" >> "$OUT"
        return
    fi
    echo "$line,$run,\"$cmd\"" >> "$OUT"
    echo "  $line"
}

for n in $SIZES; do
    echo "=== N=$n baselines ==="
    for r in $(seq 1 $REPEATS); do
        baseline sequential "./matrix_seq $n" "$n" 1 "$r"
    done
    for t in $THREADS 16 32; do
        for r in $(seq 1 $REPEATS); do
            baseline openmp "./matrix_omp $n $t" "$n" "$t" "$r"
        done
    done

    echo "=== N=$n mpi ==="
    for p in $PROCS; do
        [ "$p" -gt "$n" ] && continue
        for r in $(seq 1 $REPEATS); do
            mpirun_row "$MPIRUN -np $p ./matrix_mpi $n" "$r"
        done
    done

    echo "=== N=$n mpi+omp ==="
    for p in $PROCS; do
        [ "$p" -gt "$n" ] && continue
        for t in $THREADS; do
            for r in $(seq 1 $REPEATS); do
                mpirun_row "$MPIRUN -np $p ./matrix_mpi_omp $n $t" "$r"
            done
        done
    done

    echo "=== N=$n mpi+ocl ==="
    for p in $PROCS; do
        [ "$p" -gt "$n" ] && continue
        for r in $(seq 1 $REPEATS); do
            mpirun_row "$MPIRUN -np $p ./matrix_mpi_ocl $n" "$r"
        done
    done
done

# The m2t1p baselines write their result matrix to the working directory.
rm -f output_seq.txt output_omp.txt

echo "done, $(($(wc -l < "$OUT") - 1)) rows in $OUT"
