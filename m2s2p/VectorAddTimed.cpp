/*
  SIT315 - M2.S2P - Activity 2, Q2
  The original sequential program, unchanged except for extra timestamps so I
  can see how the runtime splits between allocation, generation and addition.
  Michael Pappas

  Build: g++ -O2 -o VectorAddTimed VectorAddTimed.cpp
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
        vector[i] = rand() % 100;
    }
}

int main()
{
    unsigned long size = 100000000;

    srand(time(0));

    int *v1, *v2, *v3;

    auto start = high_resolution_clock::now();

    v1 = (int *) malloc(size * sizeof(int *));
    v2 = (int *) malloc(size * sizeof(int *));
    v3 = (int *) malloc(size * sizeof(int *));

    auto afterAlloc = high_resolution_clock::now();

    randomVector(v1, size);

    randomVector(v2, size);

    auto afterGen = high_resolution_clock::now();

    for (int i = 0; i < size; i++)
    {
        v3[i] = v1[i] + v2[i];
    }

    auto stop = high_resolution_clock::now();

    auto allocTime = duration_cast<microseconds>(afterAlloc - start).count();
    auto genTime   = duration_cast<microseconds>(afterGen - afterAlloc).count();
    auto addTime   = duration_cast<microseconds>(stop - afterGen).count();
    auto total     = duration_cast<microseconds>(stop - start).count();

    cout << "Time taken by function: " << total << " microseconds" << endl;
    cout << "  allocation: " << allocTime << " us ("
         << (100.0 * allocTime / total) << "%)" << endl;
    cout << "  generation: " << genTime << " us ("
         << (100.0 * genTime / total) << "%)" << endl;
    cout << "  addition:   " << addTime << " us ("
         << (100.0 * addTime / total) << "%)" << endl;

    return 0;
}