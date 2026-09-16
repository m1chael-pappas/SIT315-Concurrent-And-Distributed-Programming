/*
  SIT315 - M2.S2P - Activity 2
  Sequential vector addition, with the ToDo comments completed.
  Michael Pappas

  Build: g++ -O2 -o VectorAdd VectorAdd.cpp
*/

#include <iostream>
#include <cstdlib>
#include <time.h>
#include <chrono>

using namespace std::chrono;
using namespace std;

void randomVector(int vector[], int size)
{
    for (int i = 0; i < size; i++)
    {
        // fill each slot with a pseudo random value in the range 0 to 99.
        // rand() returns 0..RAND_MAX and the modulo folds it into 0..99
        vector[i] = rand() % 100;
    }
}

int main()
{
    unsigned long size = 100000000;

    // seed the generator from the current wall clock so each run
    // produces a different sequence of random numbers
    srand(time(0));

    int *v1, *v2, *v3;

    // take a high resolution timestamp before any work starts.
    // everything after this point is being measured, which includes the
    // allocation and the vector generation, not just the addition
    auto start = high_resolution_clock::now();

    // allocate the three vectors on the heap. malloc returns void* so the
    // result is cast back to int*. NOTE: sizeof(int *) is the size of a
    // pointer (8 bytes on x86-64), not the size of an int (4 bytes), so this
    // asks for twice the memory actually needed. It still works because
    // over-allocating is harmless, but sizeof(int) is what was intended
    v1 = (int *) malloc(size * sizeof(int *));
    v2 = (int *) malloc(size * sizeof(int *));
    v3 = (int *) malloc(size * sizeof(int *));

    randomVector(v1, size);

    randomVector(v2, size);

    // the actual work: walk both input vectors element by element and write
    // the pairwise sum into the output vector. Every iteration is independent
    // of every other one, which is what makes this loop safe to parallelise
    for (int i = 0; i < size; i++)
    {
        v3[i] = v1[i] + v2[i];
    }

    auto stop = high_resolution_clock::now();

    // subtract the two timestamps and convert the resulting duration into
    // whole microseconds so it can be printed as a plain number
    auto duration = duration_cast<microseconds>(stop - start);

    cout << "Time taken by function: "
         << duration.count() << " microseconds" << endl;

    return 0;
}