// Multi-threaded vector addition with std::thread, for comparison with the
// OpenCL version. Splits the vector into one contiguous chunk per thread.
//
// Build: g++ -O2 -pthread vector_add_threads.cpp -o vector_add_threads
// Run:   ./vector_add_threads [size] [threads]
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <thread>
#include <vector>

int SZ = 8;
int THREADS = 4;
int *v1, *v2, *v3;

void init(int *&A, int size)
{
    A = (int *)malloc(sizeof(int) * size);
    for (long i = 0; i < size; i++)
        A[i] = rand() % 100;
}

void add_range(int lo, int hi)
{
    for (int i = lo; i < hi; i++)
        v3[i] = v1[i] + v2[i];
}

int main(int argc, char **argv)
{
    if (argc > 1)
        SZ = atoi(argv[1]);
    if (argc > 2)
        THREADS = atoi(argv[2]);

    init(v1, SZ);
    init(v2, SZ);
    v3 = (int *)malloc(sizeof(int) * SZ);
    // Touch the output once so page faults on first write do not land in
    // whichever run happens to go first.
    memset(v3, 0, sizeof(int) * SZ);

    // Sequential baseline.
    auto s0 = std::chrono::high_resolution_clock::now();
    add_range(0, SZ);
    auto s1 = std::chrono::high_resolution_clock::now();

    // Threaded run.
    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<std::thread> pool;
    int chunk = (SZ + THREADS - 1) / THREADS;
    for (int t = 0; t < THREADS; t++)
    {
        int lo = t * chunk;
        int hi = std::min(SZ, lo + chunk);
        if (lo < hi)
            pool.emplace_back(add_range, lo, hi);
    }
    for (auto &th : pool)
        th.join();
    auto t1 = std::chrono::high_resolution_clock::now();

    for (long i = 0; i < SZ; i++)
        if (v3[i] != v1[i] + v2[i])
        {
            printf("MISMATCH at %ld\n", i);
            return 1;
        }

    auto us = [](auto a, auto b) {
        return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count();
    };
    printf("size=%d threads=%d sequential=%ld us threaded=%ld us\n",
           SZ, THREADS, us(s0, s1), us(t0, t1));

    free(v1);
    free(v2);
    free(v3);
}
