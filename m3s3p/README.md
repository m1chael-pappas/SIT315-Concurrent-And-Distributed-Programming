# m3s3p - OpenCL

Module 3, seminar 3 practical.
Two OpenCL kernels driven from a C host program, with a `std::thread` version of the same vector addition alongside for comparison.

## Files

| File | What it does |
|---|---|
| `vector_ops.cl` | Two kernels. `square_magnitude` squares a vector in place, `vector_add` adds two vectors. |
| `vector_ops.cpp` | The host program. Picks a device, builds the kernels at runtime, runs both and verifies the result. |
| `vector_add_threads.cpp` | The CPU comparison. Runs the same addition sequentially and across `std::thread`. |
| `bench.sh` | Builds both, sweeps four sizes, prints a markdown table. |

## Build

```
g++ -O2 vector_ops.cpp -o vector_ops -lOpenCL
g++ -O2 -pthread vector_add_threads.cpp -o vector_add_threads
```

## Run

```
./vector_ops [size]                      # default 8
./vector_add_threads [size] [threads]    # defaults 8 and 4
```

`vector_ops` reads `./vector_ops.cl` from the current working directory at runtime, so run it from this folder.
It compiles the kernel source with `clBuildProgram` on each launch and dumps the build log if that fails.

Device selection takes the first platform, asks for a GPU and falls back to CPU, printing `GPU not found` when it does.

With more than 15 elements the printer shows the first five and last five values.
The final line is the one worth scraping:

```
size=1000000 kernel=812 us copy+kernel+read=4133 us
```

`kernel` is the enqueue and wait alone.
`copy+kernel+read` adds buffer creation, the host-to-device upload and the read-back, which is what a GPU speedup has to beat.

`vector_add_threads` prints one line and nothing else:

```
size=1000000 threads=16 sequential=2871 us threaded=498 us
```

Both programs verify their output by exact integer comparison and exit 1 on a mismatch.

## Benchmark

```
./bench.sh
THREADS=8 RUNS=11 ./bench.sh
```

Builds both programs, then sweeps 100000, 1000000, 10000000 and 50000000 elements.
`THREADS` defaults to `nproc` and `RUNS` to 7, and the median of each run set goes into a markdown table on stdout.
Nothing is written to disk beyond the two binaries.

Data is generated with `rand() % 100` and there is no `srand()` call, so every run produces the same vectors.
