#!/bin/bash
# SIT315 Seminar 7 - Activity 2 timing runs.
# Run from the folder holding VectorAdd.cpp and VectorAddMPI.cpp.
# Cluster runs only happen if a file called hostfile is in the same folder
# (the MPI binary must already be at the same path on the worker).
#
#   chmod +x run_mpi.sh
#   ./run_mpi.sh "1 2 4 8 16"      # on WSL2, local only
#   ./run_mpi.sh "1 2 4"           # on the master VM, local + cluster
#
# Writes results.csv with one line per run:
#   mode,procs,size,generate_us,scatter_us,add_us,gather_us,reduce_us,total_us

RUNS=5
SIZE=100000000
OUT=results.csv
PROCS=${1:-"1 2 4"}

g++ -O2 VectorAdd.cpp -o VectorAdd
mpic++ -O2 VectorAddMPI.cpp -o VectorAddMPI

echo "mode,procs,size,generate_us,scatter_us,add_us,gather_us,reduce_us,total_us" > $OUT

# Sequential baseline (the given program only reports one total time).
for r in $(seq 1 $RUNS); do
    t=$(./VectorAdd | grep -o '[0-9]*')
    echo "sequential,1,$SIZE,,,,,,$t" >> $OUT
    echo "sequential run $r: $t us"
done

# MPI on the master VM only (all ranks local).
for np in $PROCS; do
    for r in $(seq 1 $RUNS); do
        line=$(mpirun -np $np ./VectorAddMPI $SIZE | grep '^CSV' | cut -d, -f2-)
        echo "mpi-local,$line" >> $OUT
        echo "mpi-local np=$np run $r: $line"
    done
done

# MPI across both VMs via the hostfile.
[ -f hostfile ] || { echo "no hostfile, skipping cluster runs"; echo "done, results in $OUT"; exit 0; }
for np in 2 4 8; do
    for r in $(seq 1 $RUNS); do
        line=$(mpirun -np $np --hostfile hostfile ./VectorAddMPI $SIZE | grep '^CSV' | cut -d, -f2-)
        echo "mpi-cluster,$line" >> $OUT
        echo "mpi-cluster np=$np run $r: $line"
    done
done

echo "done, results in $OUT"