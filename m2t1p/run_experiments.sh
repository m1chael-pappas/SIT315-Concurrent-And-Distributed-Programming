#!/bin/bash
# M2.T1P experiment runner.
# Edit SIZES and THREADS to suit your machine, then: bash run_experiments.sh
# Results land in results.csv (5 repeats per setting so you can report the mean).

SIZES="500 1000 2000"          # pick sizes where runtime clearly grows
THREADS="2 4 8 16 32"          # 2 up to the 13900KF's 32 hardware threads
REPEATS=5

echo "version,N,threads,run,seconds,checksum" > results.csv

extract() { # pulls time and checksum out of program output
    echo "$1" | sed -E 's/.*time=([0-9.]+) s checksum=([0-9]+)/\1,\2/'
}

for N in $SIZES; do
    for r in $(seq 1 $REPEATS); do
        out=$(./matrix_seq $N)
        echo "sequential,$N,1,$r,$(extract "$out")" >> results.csv
        echo "$out"
    done
    for T in $THREADS; do
        for r in $(seq 1 $REPEATS); do
            out=$(./matrix_pthread $N $T)
            echo "pthread,$N,$T,$r,$(extract "$out")" >> results.csv
            echo "$out"
            out=$(./matrix_omp $N $T)
            echo "openmp,$N,$T,$r,$(extract "$out")" >> results.csv
            echo "$out"
        done
    done
done

echo ""
echo "Done. Results in results.csv"