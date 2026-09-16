# m2t1p - OpenMP and pthreads

Module 2, task 1.
Two exercises.
`VectorAddOMP.cpp` compares four ways of doing a parallel sum in OpenMP, and the three `matrix_*.cpp` programs multiply N x N matrices sequentially, with pthreads and with OpenMP so the three can be timed against each other.

## Files

| File | What it does |
|---|---|
| `VectorAddOMP.cpp` | Vector addition plus three reduction strategies: `atomic`, `reduction(+:)`, and per-thread partials behind `critical`. |
| `hello_mpi.cpp` | MPI hello world from the seminar. Prints rank, size and hostname. |
| `matrix_seq.cpp` | Sequential N x N integer matrix multiply, the baseline. |
| `matrix_pthread.cpp` | Same multiply, row-block decomposition across POSIX threads. |
| `matrix_omp.cpp` | Same multiply, `#pragma omp parallel for schedule(static)` on the outer loop. |
| `run_experiments.sh` | Sweeps sizes and thread counts, writes `results.csv`. |
| `results.csv` | Columns are `version,N,threads,run,seconds,checksum`. |

## Build

```
g++ -O2 -fopenmp -o VectorAddOMP VectorAddOMP.cpp
g++ -O2 -o matrix_seq matrix_seq.cpp
g++ -O2 -pthread -o matrix_pthread matrix_pthread.cpp
g++ -O2 -fopenmp -o matrix_omp matrix_omp.cpp
mpic++ -o hello_mpi hello_mpi.cpp
```

`run_experiments.sh` does not build anything, so compile first.

## Run

```
./VectorAddOMP [threads]          # default 4, size fixed at 100000000
./matrix_seq [N]                  # default N=1000
./matrix_pthread [N] [threads]    # defaults 1000 and 4
./matrix_omp [N] [threads]        # defaults 1000 and 4
mpirun -np 4 ./hello_mpi
```

The matrix programs print one line each and write their result matrix to `output_seq.txt`, `output_pthread.txt` or `output_omp.txt`.

```
version=openmp N=2000 threads=32 time=0.797563 s checksum=19590784644703
```

Matrix initialisation is seeded with `SEED = 42` and sits outside the timer, as does writing the output file.
Only the multiply is timed, and for the pthread version that includes `pthread_create` and `pthread_join`.
The checksum is how the three versions are shown to agree.

`VectorAddOMP.cpp` uses `schedule(runtime)`, so the schedule changes without rebuilding:

```
OMP_SCHEDULE=dynamic,1024 ./VectorAddOMP 16
```

## Benchmark

```
bash run_experiments.sh
```

No arguments.
Edit the three variables at the top to change the sweep: `SIZES="500 1000 2000"`, `THREADS="2 4 8 16 32"`, `REPEATS=5`.
It overwrites `results.csv` and echoes every run to stdout.
