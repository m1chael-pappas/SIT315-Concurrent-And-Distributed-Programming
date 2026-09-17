# m3t1p - MPI, hybrid MPI and OpenMP, hybrid MPI and OpenCL

Module 3, task 1.
The same N x N integer matrix multiply as m2t1p, distributed three ways.

## Files

| File | What it does |
|---|---|
| `matmul_common.h` | Row decomposition, input generation, the two collectives, the correctness check and the output format. The three programs differ only in how a rank multiplies its own band, so everything else lives here once. |
| `matrix_mpi.cpp` | MPI only. One process per band, plain triple loop inside. |
| `matrix_mpi_omp.cpp` | Hybrid. Same MPI layer, `omp parallel for` over the local rows. |
| `matrix_mpi_ocl.cpp` | Hybrid. Same MPI layer, one OpenCL kernel launch per rank over its band. |
| `matmul.cl` | The kernel. One work-item per element of C, read from the working directory at run time. |
| `run_experiments.sh` | Sweeps N, process count and threads per process. Builds the m2t1p baselines too and writes `results.csv`. |
| `results.csv` | One row per run. |

## Requirements

```
sudo apt install g++ libomp-dev opencl-headers ocl-icd-opencl-dev pocl-opencl-icd
```

MPI comes from the Anaconda install, which provides MPICH 4.3.2.
The system OpenMPI and system MPICH both hang in `MPI_Init` on this WSL2 setup, so neither is usable here.
Anaconda's `mpic++` wrapper calls a conda compiler that is not installed, so `MPICH_CXX` has to point it at the system compiler.

`pocl-opencl-icd` supplies the only OpenCL device on this machine. Any other working ICD will do.

## Build

```
export MPICH_CXX=g++
export MPI_BIN=~/anaconda3/bin

$MPI_BIN/mpic++ -O2 -o matrix_mpi matrix_mpi.cpp
$MPI_BIN/mpic++ -O2 -fopenmp -o matrix_mpi_omp matrix_mpi_omp.cpp
$MPI_BIN/mpic++ -O2 -o matrix_mpi_ocl matrix_mpi_ocl.cpp -lOpenCL
```

The path is absolute on purpose.
A bare `mpic++` resolves by PATH, and a shell without Anaconda on PATH picks up the system OpenMPI wrapper instead, which produces a binary the MPICH launcher cannot run.

The baselines from the earlier task, which `run_experiments.sh` also builds:

```
g++ -O2 -o matrix_seq ../m2t1p/matrix_seq.cpp
g++ -O2 -fopenmp -o matrix_omp ../m2t1p/matrix_omp.cpp
```

## Run

```
$MPI_BIN/mpirun -np 4 ./matrix_mpi 1000
$MPI_BIN/mpirun -np 4 ./matrix_mpi_omp 1000 8      # N then threads per process
$MPI_BIN/mpirun -np 4 ./matrix_mpi_ocl 1000
```

N defaults to 1000 and threads to 4.
N does not have to divide evenly by the process count.
The process count may not exceed N, since the split is one band per process.
`matrix_mpi_ocl` reads `./matmul.cl` at run time, so run it from this folder.

## Output

Rank 0 writes a banner and a line per phase to stderr, so a run is never a silent terminal.
Results go to stdout: one line per rank naming the rows it owned, then a phase breakdown, then a human line and a machine-readable line.

```
[mpi] N=1000 procs=16 threads=1 linked against MPICH Version:      4.3.2
  generating inputs
  broadcasting B and scattering A
  multiplying
  gathering C
rank 0 on BattleStation computed rows 0..62 (63 rows)
...
version=mpi N=1000 procs=4 threads=1 time=0.124011 s checksum=2441566030372 verify=ok
generate: 22773 us  distribute: 4449 us  compute: 118774 us  gather: 797 us
CSV,mpi,1000,4,1,22773,4449,118774,797,0.124011,2441566030372,ok
```

`time` is distribute plus compute plus gather.
Generating the inputs sits outside it, the same way m2t1p keeps initialisation out of its timer.

The banner names the MPI library the binary is actually linked against, which is the first thing to check if a run misbehaves.
Phase lines go to stderr so stdout stays clean for scraping; `run_experiments.sh` discards stderr.

`matrix_mpi_omp` also prints the MPI thread level it was given.
`matrix_mpi_ocl` prints the device it selected and adds a line splitting the device round trip into upload, kernel and read, with the one-off context and program build reported separately as untimed setup.

## How the work splits

One dimensional row decomposition.

```
                 rank 0        rank 1        rank 2        rank 3
   A  rows   [   0..249  ] [ 250..499  ] [ 500..749  ] [ 750..999  ]   MPI_Scatterv
   B  whole  [  all NxN  ] [  all NxN  ] [  all NxN  ] [  all NxN  ]   MPI_Bcast
   C  rows   [   0..249  ] [ 250..499  ] [ 500..749  ] [ 750..999  ]   MPI_Gatherv
```

Computing any element of row i of C needs all of row i of A and every column of B.
A can therefore be cut into bands and B cannot, so B goes to every rank whole.
Each rank owns a contiguous band of C, nothing is computed twice, and no rank needs data another rank holds.

| Call | Role |
|---|---|
| `MPI_Bcast` | B to every rank, N x N ints |
| `MPI_Scatterv` | A in row bands, N x N ints in total |
| `MPI_Gatherv` | C back in row bands, N x N 64-bit values in total |
| `MPI_Barrier` | fences the phases so rank 0's `MPI_Wtime` stamps bracket real work |

`Scatterv` and `Gatherv` rather than the fixed-size forms, because the variable forms take a per rank count. With `MPI_Scatter`, N would have to divide evenly by the process count. The first `N % P` ranks take one extra row.

Cost model, with P processes: compute per rank falls as N cubed over P, the scatter and gather each move N squared elements in total regardless of P, and the broadcast moves N squared elements to every rank, so it grows with P.

The hybrid versions change none of this. They change only what a rank does with its band once it has it.

`matrix_mpi_omp` initialises with `MPI_Init_thread` at `MPI_THREAD_FUNNELED`, which is sufficient because every MPI call happens outside the OpenMP parallel region on the main thread.

`matrix_mpi_ocl` gives every rank its own context, queue and program, built from `matmul.cl` at launch. The kernel index space is two dimensional, rows by columns, and both dimensions are bounds checked because the runtime may round the global size up to a multiple of the work-group size.

## Correctness

Up to N = 1000, rank 0 recomputes the whole product sequentially and compares it against the gathered result element by element, then prints `verify=ok`.
Above that the check costs more than the run itself, so it prints `checksum-only`.
The threshold is `VERIFY_MAX` in `matmul_common.h`.

The checksum is the sum of every element of C.
Inputs are generated with `SEED = 42` and `rand() % 100` in the same order as `m2t1p/matrix_seq.cpp`, so the checksum is comparable across both tasks and all five programs.

## If a run appears to hang

A working run prints its banner within a second. An empty terminal means the ranks died before reaching `MPI_Init`, and the usual cause is a binary built by one MPI and launched by another's `mpirun`.

```
ldd ./matrix_mpi | grep -oE 'libmpi[_a-z]*\.so\.[0-9]+'
```

`libmpi.so.12` is MPICH and is correct here.
`libmpi.so.40` is OpenMPI, which means the build used `/usr/bin/mpic++`; rebuild with the absolute path above.

For reference, expected wall times on the machine this was developed on, which includes roughly one second of MPI startup:

| Run | Multiply | Wall |
|---|---|---|
| `-np 16 ./matrix_mpi 1000` | 0.09 s | about 3 s, of which 1.7 s is the sequential verification |
| `-np 16 ./matrix_mpi 2000` | 1.24 s | about 4 s, no verification above N = 1000 |

## Benchmark

```
./run_experiments.sh
SIZES="500 1000" PROCS="1 4" THREADS="2 4" REPEATS=1 ./run_experiments.sh
```

| Variable | Default |
|---|---|
| `SIZES` | `100 500 1000 2000` |
| `PROCS` | `1 2 4 8 16` |
| `THREADS` | `1 2 4 8` |
| `REPEATS` | `3` |
| `OUT` | `results.csv` |
| `MPI_BIN` | `$HOME/anaconda3/bin` |

The defaults are 444 runs and take about half an hour.
The script builds all five programs first, then echoes every run as it goes.
The m2t1p baselines write their result matrix to the working directory; the script deletes those files at the end.

`results.csv` columns:

```
version,N,procs,threads,generate_us,distribute_us,compute_us,gather_us,seconds,checksum,verify,run,command
```

The last column holds the exact command that produced the row.
The three MPI programs fill the phase columns; the two m2t1p baselines print only a total, so their phase columns are empty.
