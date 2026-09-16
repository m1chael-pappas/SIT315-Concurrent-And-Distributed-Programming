# m2t3d - Traffic control simulator

Module 2, task 3.
A producer-consumer pipeline over a bounded buffer.
Producers parse a traffic CSV, consumers accumulate per-hour car counts, and the program ranks the busiest traffic lights for each hour.

One binary, three modes.
`gen` makes a dataset, `seq` is the single-threaded baseline, `par` runs the pipeline.

## Files

| File | What it does |
|---|---|
| `traffic_sim.cpp` | Everything. Generator, sequential baseline, parallel pipeline, bounded buffer, ranking. |
| `run_benchmarks.sh` | Generates the datasets if missing, then sweeps 13 configurations. Writes `results.csv`. |
| `show_results.py` | Median, spread and speedup tables from `results.csv`. Stdlib only. |
| `sample_input.csv` | 5760 rows, 20 lights over 24 hours. Small enough to read. |
| `seq.out`, `par.out` | Ranking output from a `seq` run and a `par` run, kept as the correctness proof that the two agree. |

## Build

```
g++ -std=c++17 -O2 -pthread -Wall -o traffic_sim traffic_sim.cpp
```

`run_benchmarks.sh` builds it if `./traffic_sim` is missing.

## Input format

No header row.
Three columns, `YYYY-MM-DD HH:MM,TLxxxx,cars`, with timestamps on 5-minute boundaries and light ids zero-padded to 4 digits.

```
2026-08-01 00:00,TL0000,24
```

Rows are ordered hour first, then 5-minute slot, then light id.
That ordering matters, because `probeFile()` reads only the first and last line to work out how many hours and lights the file holds instead of making a pre-pass over it.

## Run

```
traffic_sim gen --out FILE --lights N --hours H [--seed S]
traffic_sim seq --in FILE [--top N] [--print-hours K] [--out-summary FILE] [--csv]
traffic_sim par --in FILE [--producers P] [--consumers C] [--buffer B]
                [--block R] [--summary local|atomic|shared]
                [--top N] [--print-hours K] [--out-summary FILE] [--progress] [--csv]
```

| Flag | Default | Meaning |
|---|---|---|
| `--producers` | 4 | producer threads, each owning a byte range of the file |
| `--consumers` | 4 | consumer threads, also the merge thread count |
| `--buffer` | 64 | queue capacity in blocks |
| `--block` | 4096 | records per block |
| `--summary` | `local` | accumulation strategy, see below |
| `--top` | 5 | lights ranked per hour |
| `--print-hours` | 3 | hours printed to console, 0 for silent |
| `--progress` | off | monitor thread printing queue depth every 5 s |
| `--csv` | off | emit the machine-readable line |

Unknown flags are a hard error.
The generator defaults to seed 42, so datasets are reproducible.

```
./traffic_sim gen --out data/traffic_small.csv --lights 500 --hours 336
./traffic_sim par --in data/traffic_small.csv --producers 8 --consumers 8 --csv
```

`par` reports records produced and consumed, blocks pushed and popped, how many times producers waited on a full queue, how many times consumers waited on an empty one, and peak queue depth against capacity.
Those wait counts are the diagnostic when a configuration underperforms.

## The bounded buffer

A fixed-capacity `std::queue<Block>` behind one `std::mutex` and two condition variables, `notFull_` and `notEmpty_`.
Both waits sit in `while` loops so a spurious wakeup re-checks the predicate.

Shutdown works by having main join the producers, then call `close()`, which sets a flag and wakes every blocked consumer.
A consumer that finds the queue empty while closed returns false and exits.
`stop()` is the separate abort path for a parse failure.

## Three accumulation strategies

Totals live in a dense array indexed `hour * lights + light`, so there is no hashing or allocation on the hot path.
`--summary` picks how consumers write into it, and all three are kept so the report can measure the difference.

`local` gives each consumer a private array, merged after the joins across disjoint cell ranges with no locking.
`atomic` shares one array of `std::atomic<uint64_t>` with relaxed `fetch_add`.
`shared` puts one plain array behind a single mutex and locks once per record.

## Benchmark

```
./run_benchmarks.sh                    # full sweep
RUNS=5 ./run_benchmarks.sh             # override repetitions, default 3
./run_benchmarks.sh small medium       # named workloads only
```

Workloads are `small` (500 lights, 336 hours), `medium` (1000 lights, 1680 hours) and `large` (1500 lights, 3360 hours).
Missing datasets are generated on first run.
Each workload gets a discarded warm-up pass first, otherwise the first timed run pays for the cold read off disk and the comparison measures storage rather than threading.

## Regenerating the datasets

`data/` is gitignored, 5.1 GB across four files.
Running `./run_benchmarks.sh` regenerates the three it needs.
Manually:

```
./traffic_sim gen --out data/traffic_small.csv  --lights 500  --hours 336
./traffic_sim gen --out data/traffic_medium.csv --lights 1000 --hours 1680
./traffic_sim gen --out data/traffic_large.csv  --lights 1500 --hours 3360
./traffic_sim gen --out data/traffic_xlarge.csv --lights 2000 --hours 5040
```

`traffic_xlarge.csv` at 3.0 GB is a manual stress test.
It has no entry in `run_benchmarks.sh`, so passing `xlarge` to the script fails on the array lookup.

`sample_input.csv` comes from `--lights 20 --hours 24`.

`seq.out` and `par.out` are not produced by any script:

```
./traffic_sim seq --in data/traffic_small.csv --top 5 --print-hours 0 --out-summary seq.out
./traffic_sim par --in data/traffic_small.csv --producers 8 --consumers 8 --top 5 --print-hours 0 --out-summary par.out
diff seq.out par.out
```
