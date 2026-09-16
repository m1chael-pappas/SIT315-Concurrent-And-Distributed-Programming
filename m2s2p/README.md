# m2s2p - Vector addition with std::thread

Module 2, seminar 2 practical.
Takes the unit's sequential vector addition program, splits its timer to show where the time goes, then parallelises it with `std::thread`.

Problem size is 100,000,000 `int` elements.

## Files

| File | What it does |
|---|---|
| `VectorAdd.cpp` | The starter program, unmodified. Sequential, one timer around everything. |
| `VectorAddTimed.cpp` | Same program with the timer split into allocation, generation and addition. |
| `VectorAddParallel.cpp` | Parallel version. Static block and dynamic chunk decomposition, plus parallel generation. |
| `benchmark.sh` | Thread sweep, partition sweep and a static-vs-dynamic comparison. |

## Build

```
g++ -O2 -o VectorAdd VectorAdd.cpp
g++ -O2 -o VectorAddTimed VectorAddTimed.cpp
g++ -O2 -std=c++17 -pthread -o VectorAddParallel VectorAddParallel.cpp
```

## Run

`VectorAdd` and `VectorAddTimed` take no arguments.

```
./VectorAddParallel [size] [threads] [partition] [mode]
```

| Argument | Default | Meaning |
|---|---|---|
| `size` | 100000000 | element count |
| `threads` | `hardware_concurrency()` | worker threads, clamped to at least 1 |
| `partition` | 0 | elements per work unit. 0 means one even block per thread, or 65536 in dynamic mode |
| `mode` | 0 | 0 for static block, 1 for dynamic chunks off a shared counter |

Running with `1 0 0` gives the sequential baseline through the same code path.

Output is one line of key-value fields, and the exit code is 1 if the correctness sample fails.

```
size=100000000 threads=16 partition=6250000 mode=block gen_us=91633 add_us=24118 total_us=115751 correct=yes
```

## Benchmark

```
./benchmark.sh [size] [repeats]
```

Defaults to the full size and 3 repeats.
Rebuilds `VectorAddParallel`, then prints three markdown tables to stdout: a thread sweep from 1 to 64 in static mode, a partition sweep from 1024 to 33554432 in dynamic mode, and static against dynamic at `nproc` threads.
Each configuration runs `repeats` times and the best time is kept.
Nothing is written to disk beyond the rebuilt binary.
