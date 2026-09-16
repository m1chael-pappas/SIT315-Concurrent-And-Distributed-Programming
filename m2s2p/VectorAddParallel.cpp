/*
  SIT315 - M2.S2P - Activity 2
  Parallel vector addition using std::thread.
  Michael Pappas

  Build: g++ -O2 -std=c++17 -pthread -o VectorAddParallel VectorAddParallel.cpp

  Usage:
    ./VectorAddParallel [size] [threads] [partition] [mode]

      size       number of elements            (default 100000000)
      threads    number of worker threads      (default hardware concurrency)
      partition  elements per work unit        (default 0, meaning one even
                                                block per thread)
      mode       0 = static block, 1 = dynamic chunks off a shared counter

  Passing 1 thread in static block mode gives the sequential baseline, so the
  comparison is apples to apples.
*/

#include <iostream>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <thread>
#include <vector>
#include <atomic>
#include <random>

using namespace std::chrono;
using namespace std;

// ---------------------------------------------------------------------------
// Vector generation
// ---------------------------------------------------------------------------

// Each thread gets its own mt19937 engine. I cannot use rand() here because it
// keeps hidden global state, so calling it from several threads at once is a
// data race and also serialises on an internal lock in glibc.
void fillRange(int *v1, int *v2, unsigned long from, unsigned long to,
               unsigned int seed)
{
    mt19937 engine(seed);
    uniform_int_distribution<int> dist(0, 99);

    for (unsigned long i = from; i < to; i++)
    {
        v1[i] = dist(engine);
        v2[i] = dist(engine);
    }
}

// ---------------------------------------------------------------------------
// Static block decomposition
// ---------------------------------------------------------------------------

// Thread t owns one contiguous slice and never touches anyone else's slice,
// so no locking is needed anywhere in here.
void addBlock(const int *v1, const int *v2, int *v3,
              unsigned long from, unsigned long to)
{
    for (unsigned long i = from; i < to; i++)
    {
        v3[i] = v1[i] + v2[i];
    }
}

// ---------------------------------------------------------------------------
// Dynamic chunk decomposition
// ---------------------------------------------------------------------------

// All threads pull work units off one shared atomic counter. fetch_add hands
// each caller a unique starting index without a mutex. This is how I vary the
// partition size independently of the thread count.
void addDynamic(const int *v1, const int *v2, int *v3,
                unsigned long size, unsigned long partition,
                atomic<unsigned long> *nextIndex)
{
    while (true)
    {
        unsigned long from = nextIndex->fetch_add(partition);

        if (from >= size)
        {
            break;
        }

        unsigned long to = from + partition;
        if (to > size)
        {
            to = size;
        }

        for (unsigned long i = from; i < to; i++)
        {
            v3[i] = v1[i] + v2[i];
        }
    }
}

// ---------------------------------------------------------------------------

int main(int argc, char **argv)
{
    unsigned long size      = (argc > 1) ? strtoul(argv[1], nullptr, 10) : 100000000UL;
    unsigned int  threads   = (argc > 2) ? (unsigned int) atoi(argv[2]) : thread::hardware_concurrency();
    unsigned long partition = (argc > 3) ? strtoul(argv[3], nullptr, 10) : 0UL;
    int           mode      = (argc > 4) ? atoi(argv[4]) : 0;

    if (threads < 1)
    {
        threads = 1;
    }

    // in static block mode the partition size is just the slice each thread
    // gets, so it is derived from the thread count rather than chosen
    if (mode == 0)
    {
        partition = (size + threads - 1) / threads;
    }
    else if (partition == 0)
    {
        partition = 65536;
    }

    int *v1 = (int *) malloc(size * sizeof(int));
    int *v2 = (int *) malloc(size * sizeof(int));
    int *v3 = (int *) malloc(size * sizeof(int));

    if (v1 == nullptr || v2 == nullptr || v3 == nullptr)
    {
        cerr << "allocation failed for size " << size << endl;
        return 1;
    }

    // -----------------------------------------------------------------------
    // Vector generation phase: fill the two input vectors with random values in parallel
    // -----------------------------------------------------------------------
    auto genStart = high_resolution_clock::now();

    {
        vector<thread> pool;
        unsigned long slice = (size + threads - 1) / threads;

        for (unsigned int t = 0; t < threads; t++)
        {
            unsigned long from = t * slice;
            unsigned long to   = from + slice;

            if (from >= size)
            {
                break;
            }
            if (to > size)
            {
                to = size;
            }

            // seed differs per thread so the slices are not identical
            pool.emplace_back(fillRange, v1, v2, from, to, 1234u + t);
        }

        for (auto &th : pool)
        {
            th.join();
        }
    }

    auto genStop = high_resolution_clock::now();

    // -----------------------------------------------------------------------
    //  Addition phase: add the two input vectors into the output vector in parallel
    // -----------------------------------------------------------------------
    auto addStart = high_resolution_clock::now();

    if (mode == 0)
    {
        vector<thread> pool;

        for (unsigned int t = 0; t < threads; t++)
        {
            unsigned long from = t * partition;
            unsigned long to   = from + partition;

            if (from >= size)
            {
                break;
            }
            if (to > size)
            {
                to = size;
            }

            pool.emplace_back(addBlock, v1, v2, v3, from, to);
        }

        for (auto &th : pool)
        {
            th.join();
        }
    }
    else
    {
        atomic<unsigned long> nextIndex(0);
        vector<thread> pool;

        for (unsigned int t = 0; t < threads; t++)
        {
            pool.emplace_back(addDynamic, v1, v2, v3, size, partition, &nextIndex);
        }

        for (auto &th : pool)
        {
            th.join();
        }
    }

    auto addStop = high_resolution_clock::now();

    // -----------------------------------------------------------------------
    // Correctness check
    // -----------------------------------------------------------------------
    bool correct = true;
    unsigned long step = (size / 1000) + 1;

    for (unsigned long i = 0; i < size; i += step)
    {
        if (v3[i] != v1[i] + v2[i])
        {
            correct = false;
            break;
        }
    }

    if (size > 0 && v3[size - 1] != v1[size - 1] + v2[size - 1])
    {
        correct = false;
    }

    auto genTime   = duration_cast<microseconds>(genStop - genStart).count();
    auto addTime   = duration_cast<microseconds>(addStop - addStart).count();
    auto totalTime = genTime + addTime;

    cout << "size="      << size
         << " threads="  << threads
         << " partition=" << partition
         << " mode="     << (mode == 0 ? "block" : "dynamic")
         << " gen_us="   << genTime
         << " add_us="   << addTime
         << " total_us=" << totalTime
         << " correct="  << (correct ? "yes" : "NO")
         << endl;

    free(v1);
    free(v2);
    free(v3);

    return correct ? 0 : 1;
}