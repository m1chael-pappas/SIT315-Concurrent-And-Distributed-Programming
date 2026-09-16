#include <iostream>
#include <cstdlib>
#include <cstring>
#include <time.h>
#include <chrono>
#include <omp.h>

using namespace std::chrono;
using namespace std;

// Activities 1 and 2 
void randomVector(int vector[], int size)
{
    for (int i = 0; i < size; i++)
    {
        // Fill each element with a random value between 0 and 99
        vector[i] = rand() % 100;
    }
}


int main(int argc, char* argv[]){

    unsigned long size = 100000000;

    // Optional thread count from the command line, defaults to 4
    int threads = (argc > 1) ? atoi(argv[1]) : 4;
    omp_set_num_threads(threads);

    srand(time(0));

    int *v1, *v2, *v3;

    // Start the overall timer, same position as the provided program
    auto start = high_resolution_clock::now();

    // Allocate the three vectors on the heap
    v1 = (int *) malloc(size * sizeof(int *));
    v2 = (int *) malloc(size * sizeof(int *));
    v3 = (int *) malloc(size * sizeof(int *));


    randomVector(v1, size);

    randomVector(v2, size);

    // Touch v3 before the timer so the addition loop is not the first write
    // to it. Without this, one thread pays the cost of page faulting 400 MB
    // of fresh pages inside the timed section and the single thread baseline
    // is unreliable (identified in my M2.S2P analysis).
    memset(v3, 0, size * sizeof(int));

    // ------------------------------------------------------------------
    // Activity 1 / Activity 2 part 1: parallel addition.
    // omp parallel creates the team of threads, omp for splits the loop
    // iterations between them. Every iteration writes a different v3[i],
    // so no synchronisation is needed.
    // default(none) turns off the default data sharing rules and forces
    // every variable used inside the region to be listed. Without the
    // shared(...) clause this does not compile. The vectors and size must
    // be shared so all threads work on the same real arrays. The loop
    // index is automatically private.
    // ------------------------------------------------------------------
    auto addStart = high_resolution_clock::now();

    #pragma omp parallel default(none) shared(size, v1, v2, v3)
    {
        #pragma omp for schedule(runtime)
        for (int i = 0; i < size; i++)
        {
            v3[i] = v1[i] + v2[i];
        }
    }

    auto stop = high_resolution_clock::now();

    // Total duration measured from before allocation, matching the provided
    // program, plus the addition loop on its own
    auto duration = duration_cast<microseconds>(stop - start);
    auto addDuration = duration_cast<microseconds>(stop - addStart);

    // ------------------------------------------------------------------
    // Activity 2 part 2: total sum using a shared variable and atomic
    // update. Every iteration synchronises on the same variable.
    // ------------------------------------------------------------------
    long long total = 0;
    auto t0 = high_resolution_clock::now();
    #pragma omp parallel default(none) shared(size, v3, total)
    {
        #pragma omp for schedule(runtime)
        for (int i = 0; i < size; i++)
        {
            #pragma omp atomic
            total += v3[i];
        }
    }
    auto atomicTime = duration_cast<microseconds>(high_resolution_clock::now() - t0);
    long long totalAtomic = total;

    // ------------------------------------------------------------------
    // Activity 2 part 3: same sum using the reduction clause. Each thread
    // gets its own private copy of total and OpenMP combines them at the
    // end. No contention during the loop.
    // ------------------------------------------------------------------
    total = 0;
    t0 = high_resolution_clock::now();
    #pragma omp parallel for reduction(+:total) schedule(runtime)
    for (int i = 0; i < size; i++)
    {
        total += v3[i];
    }
    auto reductionTime = duration_cast<microseconds>(high_resolution_clock::now() - t0);
    long long totalReduction = total;

    // ------------------------------------------------------------------
    // Activity 2 part 4: manual version of what reduction does. Each
    // thread sums into its own private variable, then a critical section
    // combines the partial sums, running once per thread instead of once
    // per element.
    // ------------------------------------------------------------------
    total = 0;
    t0 = high_resolution_clock::now();
    #pragma omp parallel default(none) shared(size, v3, total)
    {
        long long localSum = 0;
        #pragma omp for schedule(runtime)
        for (int i = 0; i < size; i++)
        {
            localSum += v3[i];
        }
        #pragma omp critical
        total += localSum;
    }
    auto criticalTime = duration_cast<microseconds>(high_resolution_clock::now() - t0);
    long long totalCritical = total;

    cout << "Time taken by function: "
         << duration.count() << " microseconds" << endl;
    cout << "Addition loop only (" << threads << " threads): "
         << addDuration.count() << " microseconds" << endl;
    cout << "atomic:    " << atomicTime.count() << " microseconds, total = " << totalAtomic << endl;
    cout << "reduction: " << reductionTime.count() << " microseconds, total = " << totalReduction << endl;
    cout << "critical:  " << criticalTime.count() << " microseconds, total = " << totalCritical << endl;
    cout << "totals match: "
         << ((totalAtomic == totalReduction && totalReduction == totalCritical) ? "yes" : "NO")
         << endl;

    return 0;
}