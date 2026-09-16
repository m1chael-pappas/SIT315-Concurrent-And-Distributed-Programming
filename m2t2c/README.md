# m2t2c - Parallel quicksort

Module 2, task 2.
One quicksort, three implementations, sorting up to 200 million `int32_t`.
The sequential version is the baseline, the OpenMP version parallelises the recursion with tasks, and the `std::thread` version forks real OS threads to a fixed depth.

## Files

| File | What it does |
|---|---|
| `quicksort_sequential.cpp` | Baseline. Tail-recursion elimination keeps stack depth at O(log n). |
| `quicksort_parallel_openmp.cpp` | OpenMP tasks. One thread generates tasks under `omp single`, the rest steal them. |
| `quicksort_parallel_threads.cpp` | Raw `std::thread`. Forks the left half, keeps the right half, down to `ceil(log2(threads))` levels. |
| `run_benchmarks.sh` | The full sweep. Writes `results.csv` and `bench_log.txt`. |
| `run_pivot_test.sh` | Compares the four pivot rules. Writes `pivot_results.csv`. |
| `show_results.py` | Prints speedup tables from `results.csv`. Stdlib only. |

All three share an identical core: Hoare partition, insertion sort below 32 elements, the pivot rules, dataset generation and the checksum.
Only `quicksort()` and `main()` differ.

## Build

```
g++ -O2 -std=c++17 -Wall -Wextra quicksort_sequential.cpp -o quicksort_sequential
g++ -O2 -std=c++17 -Wall -Wextra -fopenmp quicksort_parallel_openmp.cpp -o quicksort_parallel_openmp
g++ -O2 -std=c++17 -Wall -Wextra -pthread quicksort_parallel_threads.cpp -o quicksort_parallel_threads
```

Neither script builds anything, so compile first.

## Run

Flags are `--key value` and order does not matter.
Unknown flags are ignored silently and there is no `--help`.

| Flag | Default | Meaning |
|---|---|---|
| `--n` | 10000000 | elements to generate |
| `--seed` | 315 | RNG seed |
| `--pattern` | `random` | `random`, `sorted` or `reverse` |
| `--pivot` | `median3` | `median3`, `middle`, `last` or `first` |
| `--runs` | 5 | timed repetitions, reported as the median |
| `--input` | none | load raw `int32` from a file, overriding `--n` and `--pattern` |
| `--save` | none | write the generated dataset to a file as raw `int32` |

The two parallel builds add `--threads` (default 8) and a cutoff below which they stop splitting: `--task-cutoff` for OpenMP and `--size-cutoff` for `std::thread`, both defaulting to 100000.

```
./quicksort_parallel_openmp --n 200000000 --threads 32 --task-cutoff 100000
```

Every run verifies the output is sorted, prints a checksum, then emits a machine-readable line.
A run that fails either check prints `FAILED` and exits 1.

```
CSV,omp,200000000,32,median3,1502.385
```

## Benchmark

```
./run_benchmarks.sh
```

No arguments.
Sweeps 10M, 50M, 100M and 200M elements against 2, 4, 8, 16, 24 and 32 threads, 5 runs each.
Writes `results.csv` and tees full output into `bench_log.txt`.

```
python3 show_results.py results.csv
```

Prints OpenMP and `std::thread` speedups side by side plus the ratio between them.

```
./run_pivot_test.sh [N] [threads]
```

Defaults to 50000000 and 16.
Writes `pivot_results.csv`, then runs an already-sorted stress test at 3M elements with a 60 second timeout.
That last section prints to the console only and is not captured in any file.

## Regenerating the dataset

`data_200M.bin` is gitignored and no script produces it.

```
./quicksort_sequential --n 200000000 --seed 315 --save data_200M.bin --runs 1
```

The sweep does not need it.
It regenerates in memory from `--n` and `--seed`.
