# m3t2c - Distributed sort with MPI, and with MPI + OpenCL

Module 3, task 2.
The quicksort from m2t2c, distributed two ways.

Both programs run the same distributed algorithm and differ in one step: where a rank sorts its own chunk.
`quicksort_mpi` uses the m2t2c quicksort on the CPU.
`quicksort_mpi_ocl` uploads the chunk to an OpenCL device and sorts it there with a bitonic sorting network.

## Files

| File | What it does |
|---|---|
| `qsort_common.h` | Dataset generation, checksum, sorted check, timer, argument parsing and the local quicksort. Copied unchanged from m2t2c so both programs sort identical bytes and print a comparable checksum. |
| `psrs.h` | The MPI layer. Parallel sorting by regular sampling, with the local sort passed in as a callable. Shared by both programs. |
| `quicksort_mpi.cpp` | MPI only. Local sort is the m2t2c quicksort. |
| `quicksort_mpi_ocl.cpp` | MPI + OpenCL. Local sort is a bitonic network on the device. |
| `bitonic.cl` | The kernel. One compare-exchange stage per launch, read from the working directory at run time. |
| `run_experiments.sh` | Sweeps size and process count for both programs, runs the three m2t2c baselines, and sweeps the OpenCL local work size. Writes `results.csv`. |
| `results.csv` | One row per run. |

## Requirements

```
sudo apt install g++ libomp-dev opencl-headers ocl-icd-opencl-dev pocl-opencl-icd
```

MPI comes from the Anaconda install, which provides MPICH 4.3.2.
The system OpenMPI and system MPICH both hang in `MPI_Init` on this WSL2 setup, so neither is usable here.
Anaconda's `mpic++` wrapper calls a conda compiler that is not installed, so `MPICH_CXX` has to point it at the system compiler.

## Build

```
export MPICH_CXX=g++
export MPI_BIN=~/anaconda3/bin

$MPI_BIN/mpic++ -O2 -std=c++17 -Wall -Wextra -o quicksort_mpi quicksort_mpi.cpp
$MPI_BIN/mpic++ -O2 -std=c++17 -Wall -Wextra -o quicksort_mpi_ocl quicksort_mpi_ocl.cpp -lOpenCL
```

The path is absolute on purpose.
A bare `mpic++` resolves by PATH, and a shell without Anaconda on PATH picks up the system OpenMPI wrapper instead, which produces a binary the MPICH launcher cannot run.

The m2t2c baselines, which `run_experiments.sh` also builds:

```
g++ -O2 -std=c++17 -o quicksort_sequential       ../m2t2c/quicksort_sequential.cpp
g++ -O2 -std=c++17 -fopenmp -o quicksort_parallel_openmp  ../m2t2c/quicksort_parallel_openmp.cpp
g++ -O2 -std=c++17 -pthread -o quicksort_parallel_threads ../m2t2c/quicksort_parallel_threads.cpp
```

## Run

```
$MPI_BIN/mpirun -np 4 ./quicksort_mpi     --n 16000000 --runs 5
$MPI_BIN/mpirun -np 4 ./quicksort_mpi_ocl --n 16000000 --runs 5
```

Flags are `--key value` and order does not matter, matching m2t2c.
Unknown flags are ignored silently.

| Flag | Default | Meaning | Applies to |
|---|---|---|---|
| `--n` | 16000000 | elements to generate | both |
| `--seed` | 315 | RNG seed | both |
| `--pattern` | `random` | `random`, `sorted` or `reverse` | both |
| `--runs` | 5 | timed repetitions, reported as the median | both |
| `--pivot` | `median3` | `median3`, `middle`, `last` or `first` | `quicksort_mpi` |
| `--local` | 0 | OpenCL local work size, 0 lets the runtime choose | `quicksort_mpi_ocl` |

The process count must satisfy `n >= p * p`, because each rank takes `p` samples from its own chunk.
Below that the program prints the constraint and exits 1.

`quicksort_mpi_ocl` reads `./bitonic.cl` at run time, so run it from this folder.

## Output

Rank 0 writes a banner and a line per run to stderr, so a run is never a silent terminal.
Results go to stdout.

```
[mpi+ocl] n=16000000 procs=4 runs=5 local=runtime choice
      device: CPU pthread-13th Gen Intel(R) Core(TM) i9-13900KF
      linked against MPICH Version:      4.3.2
```

`quicksort_mpi_ocl` prints one line per rank naming the device that rank selected, before any sorting happens:

```
rank 0 on BattleStation using CPU device "pthread-13th Gen Intel(R) Core(TM) i9-13900KF" (The pocl project, 32 compute units, max work-group 4096)
```

Then per run, the median, the phase breakdown, the kernel configuration and a machine-readable line:

```
run 1        340.76 ms   verified   (253 launches, upload 3.1, kernel 298.4, read 1.2 ms)
median       340.76 ms
phases    scatter 4.78  localsort 304.94  pivots 0.02  exchange 7.62  merge 19.36  gather 4.04  (last run, ms)
kernel    253 launches, global work size 4194304 per launch, local runtime choice
CSV,mpi+ocl,16000000,4,bitonic,340.761,...
```

`verified` means the result is in ascending order **and** its checksum equals the checksum of the unsorted input.
A run that fails either check prints `FAILED` and aborts.

The reported time covers scatter, local sort, pivot selection, all-to-all exchange, merge and gather.
Generating the dataset sits outside it, the same convention m2t2c used.

## How the work splits

Parallel sorting by regular sampling, in `psrs.h`.

```
  1. scatter        rank 0 cuts the array into p equal contiguous chunks
  2. local sort     every rank sorts its own chunk            <- the only difference
  3. sample         every rank takes p evenly spaced samples of its sorted chunk
  4. pivots         rank 0 gathers all p*p samples, sorts them, picks p-1 pivots, broadcasts
  5. bucket         every rank splits its sorted chunk at those pivots into p buckets
  6. all-to-all     bucket i from every rank goes to rank i
  7. merge          every rank merges the p sorted runs it received
  8. gather         rank 0 collects the merged runs in rank order
```

After step 6 every element on rank r is less than or equal to every element on rank r+1.
That is what makes step 8 a plain concatenation rather than another merge, and it is the reason for the sampling: the pivots are chosen from the data itself so the buckets come out roughly even, instead of being guessed from the value range.

| Call | Role |
|---|---|
| `MPI_Scatterv` | the input in equal chunks |
| `MPI_Gather` | the `p` samples from each rank, and later the final bucket sizes |
| `MPI_Bcast` | the `p-1` pivots to every rank |
| `MPI_Alltoall` | how many elements each rank is sending each other rank |
| `MPI_Alltoallv` | the buckets themselves |
| `MPI_Gatherv` | the merged runs back to rank 0 |
| `MPI_Barrier` | fences the phases so the timestamps bracket real work |

Bucket boundaries come from `std::upper_bound` rather than a scan, because the chunk is already sorted by then.

`psrs::sort` takes the local sort as a template parameter, so the two programs share this file byte for byte and differ only in the callable they pass.

## The kernel

Bitonic sort is a sorting network: a fixed sequence of compare-and-swap pairs that does not depend on the values.
That is what makes it expressible on a device with thousands of work-items and no cheap global synchronisation.
Quicksort is not, because where it recurses next depends on where the partition landed.

The host drives two nested loops. `k` is the size of the bitonic sequence being merged and doubles from 2 to the padded length. `j` is the compare distance inside that merge and halves from `k/2` to 1.
Every `(k, j)` pair is one kernel launch, so a chunk padded to 2^m elements takes `m(m+1)/2` launches.
They cannot be fused: each launch has to see the previous one finished, and OpenCL has no barrier across work-groups.

| Setting | Value |
|---|---|
| Global work size | the padded chunk length, one work-item per element |
| Local work size | runtime choice by default, `--local` to fix it |
| Launches | `m(m+1)/2` for a chunk padded to 2^m |
| Padding | up to the next power of two, filled with `INT32_MAX` so it sorts to the tail and is dropped on read-back |

Each work-item owns index `i` and pairs with `i ^ j`, and only the lower of the two performs the swap, so exactly half the work-items are idle in every launch.
That is inherent to the network rather than a defect.

Data crossing the bus, per rank per run: the padded chunk goes up once, every launch operates on that one device buffer with nothing copied in between, and the sorted chunk comes back once.
Everything after the local sort is MPI on the host and never touches the device.

## Correctness

Rank 0 checksums the input before sorting, then after the gather checks the result is in ascending order and that its checksum is unchanged.

The checksum is the sum of every element taken as unsigned, so it wraps rather than overflowing.
Sorting is a permutation, so the value has to survive untouched.
It catches a distributed bug that drops or duplicates an element during the all-to-all, which an is-it-sorted check on its own would walk straight past.

Because `qsort_common.h` generates the dataset exactly as m2t2c does, the checksum is directly comparable against the sequential, `std::thread` and OpenMP programs from module 2 at the same `--n` and `--seed`.

## Benchmark

```
./run_experiments.sh
SIZES="1000000 4000000" PROCS="1 4" RUNS=1 ./run_experiments.sh
```

| Variable | Default |
|---|---|
| `SIZES` | `1000000 4000000 16000000 64000000` |
| `PROCS` | `1 2 4 8 16` |
| `THREADS` | `8 16 32` (for the m2t2c baselines) |
| `RUNS` | `3` |
| `LOCALS` | `0 16 32 64 128 256 1024` |
| `LOCAL_N` | `16000000` |
| `LOCAL_P` | `4` |
| `OUT` | `results.csv` |
| `MPI_BIN` | `$HOME/anaconda3/bin` |

`results.csv` columns:

```
impl,elements,procs,threads,median_ms,scatter_ms,localsort_ms,pivots_ms,exchange_ms,merge_ms,gather_ms,checksum,device,local_ws,run,command
```

The last column holds the exact command that produced the row.
The two programs here fill the phase columns; the three m2t2c baselines print only a total, so their phase columns are empty.
