# m3s2p - Distributed vector addition with MPI

Module 3, seminar 2 practical.
Two activities.
`hello_mpi_comm.cpp` sends the same message point-to-point and then collectively so the two can be compared, and `VectorAddMPI.cpp` distributes a 100,000,000 element vector addition across ranks with per-phase timing.

## Files

| File | What it does |
|---|---|
| `VectorAdd.cpp` | The sequential starter program, unmodified. The baseline. |
| `VectorAddMPI.cpp` | Distributed version. Scatter, local add, gather, reduce. |
| `hello_mpi_comm.cpp` | Activity 1. `MPI_Send`/`MPI_Recv` in a loop, then the same thing as one `MPI_Bcast`. |
| `run_mpi.sh` | Sequential baseline plus an MPI process sweep. Writes `results.csv`. |
| `vm_results/` | Captured output from runs on the unit VM, including its environment in `out_env.txt`. |

## Build

```
g++ -O2 VectorAdd.cpp -o VectorAdd
mpic++ -O2 VectorAddMPI.cpp -o VectorAddMPI
mpic++ hello_mpi_comm.cpp -o hello_mpi_comm
```

## Run

```
./VectorAdd                                    # no arguments, size fixed at 100000000
mpirun -np 4 ./VectorAddMPI [size]             # size defaults to 100000000
mpirun -np 4 ./hello_mpi_comm                  # no arguments
```

`size` must divide evenly by the process count.
`MPI_Scatter` requires equal chunks, so rank 0 prints an error and exits 1 rather than silently dropping the remainder.


## Benchmark

```
./run_mpi.sh              # defaults to "1 2 4"
./run_mpi.sh "1 2 4 8"
```

Compiles both programs, runs the sequential baseline 5 times, then runs `VectorAddMPI` 5 times per process count.
Writes `results.csv` with a `mode` column of `sequential` or `mpi-local`.

Cluster runs need a `hostfile` in this directory and the binary at the same path on every worker node.
There is no hostfile here, so the script stops after the local runs and prints `no hostfile, skipping cluster runs`.
With one present it also sweeps a hard-coded 2, 4 and 8 processes and tags those rows `mpi-cluster`.

