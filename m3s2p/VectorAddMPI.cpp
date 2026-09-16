// SIT315 Seminar 7 - Activity 2 (distributed vector addition)
// Extends the provided VectorAdd.cpp with MPI:
//   Step 1: rank 0 builds v1 and v2, MPI_Scatter hands each rank an equal
//           chunk, every rank adds its chunk, MPI_Gather assembles v3 on rank 0.
//   Step 3: every rank sums its own v3 chunk and MPI_Reduce (MPI_SUM) folds
//           the partial sums into one total on rank 0.
// Timing is printed by rank 0 so the run script can compare against the
// sequential and multi-threaded versions.
//
// Build:  mpic++ -O2 VectorAddMPI.cpp -o VectorAddMPI
// Run:    mpirun -np 4 --hostfile hostfile ./VectorAddMPI [size]

#include <mpi.h>
#include <iostream>
#include <cstdlib>
#include <time.h>

using namespace std;

void randomVector(int vector[], int size)
{
    for (int i = 0; i < size; i++)
    {
        // Fill each slot with a value in 0..99, same as the sequential version.
        vector[i] = rand() % 100;
    }
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank, numtasks, name_len;
    char name[MPI_MAX_PROCESSOR_NAME];
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Get_processor_name(name, &name_len);

    // Same default as VectorAdd.cpp; optional override from the command line.
    unsigned long size = 100000000;
    if (argc > 1) size = strtoul(argv[1], NULL, 10);

    // MPI_Scatter sends equal-sized pieces, so the size has to divide evenly.
    if (size % numtasks != 0)
    {
        if (rank == 0)
            cerr << "size " << size << " is not divisible by " << numtasks
                 << " processes" << endl;
        MPI_Finalize();
        return 1;
    }
    int chunk = size / numtasks;

    int *v1 = NULL, *v2 = NULL, *v3 = NULL;   // full vectors, rank 0 only
    int *l1, *l2, *l3;                        // this rank's chunk

    // Every rank owns just its slice of each vector.
    l1 = (int *) malloc(chunk * sizeof(int));
    l2 = (int *) malloc(chunk * sizeof(int));
    l3 = (int *) malloc(chunk * sizeof(int));

    double t_start = 0, t_gen = 0, t_scatter = 0, t_add = 0, t_gather = 0, t_reduce = 0;

    // ---------- Step 1a: rank 0 builds the input vectors ----------
    if (rank == 0)
    {
        t_start = MPI_Wtime();
        srand(time(0));
        v1 = (int *) malloc(size * sizeof(int));
        v2 = (int *) malloc(size * sizeof(int));
        v3 = (int *) malloc(size * sizeof(int));
        randomVector(v1, size);
        randomVector(v2, size);
        t_gen = MPI_Wtime();
    }

    // ---------- Step 1b: scatter the work ----------
    // Rank 0 sends chunk elements of v1 to each rank (including itself).
    // Every rank receives its piece into l1. Same again for v2 into l2.
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Scatter(v1, chunk, MPI_INT, l1, chunk, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(v2, chunk, MPI_INT, l2, chunk, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) t_scatter = MPI_Wtime();

    // ---------- Step 1c: each rank adds its own chunk ----------
    for (int i = 0; i < chunk; i++)
    {
        l3[i] = l1[i] + l2[i];
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) t_add = MPI_Wtime();

    // ---------- Step 1d: gather the results back on rank 0 ----------
    // The reverse of scatter: rank 0 collects every l3 into v3, in rank order.
    MPI_Gather(l3, chunk, MPI_INT, v3, chunk, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) t_gather = MPI_Wtime();

    // ---------- Step 3: total of v3 with MPI_Reduce ----------
    // Each rank sums its local chunk (no communication), then MPI_Reduce
    // adds the numtasks partial sums together and leaves the result on rank 0.
    long long local_sum = 0;
    for (int i = 0; i < chunk; i++)
    {
        local_sum += l3[i];
    }
    long long total_sum = 0;
    MPI_Reduce(&local_sum, &total_sum, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) t_reduce = MPI_Wtime();

    // Each rank reports where it ran, which proves the work left the master.
    cout << "rank " << rank << " on " << name << " added " << chunk
         << " elements, local sum " << local_sum << endl;

    if (rank == 0)
    {
        // Check the gathered v3 against a plain sequential pass.
        long long check_sum = 0;
        bool ok = true;
        for (unsigned long i = 0; i < size; i++)
        {
            if (v3[i] != v1[i] + v2[i]) ok = false;
            check_sum += v3[i];
        }

        cout << "processes: " << numtasks << ", size: " << size << endl;
        cout << "v3 correct: " << (ok ? "yes" : "NO") << endl;
        cout << "MPI_Reduce total: " << total_sum
             << ", sequential check: " << check_sum
             << (total_sum == check_sum ? " (match)" : " (MISMATCH)") << endl;

        // Times in microseconds so they line up with VectorAdd.cpp's output.
        auto us = [](double a, double b) { return (long)((b - a) * 1e6); };
        cout << "generate:  " << us(t_start, t_gen)     << " us" << endl;
        cout << "scatter:   " << us(t_gen, t_scatter)   << " us" << endl;
        cout << "add:       " << us(t_scatter, t_add)   << " us" << endl;
        cout << "gather:    " << us(t_add, t_gather)    << " us" << endl;
        cout << "reduce:    " << us(t_gather, t_reduce) << " us" << endl;
        cout << "total (generate+scatter+add+gather): "
             << us(t_start, t_gather) << " us" << endl;
        cout << "CSV," << numtasks << "," << size << ","
             << us(t_start, t_gen) << "," << us(t_gen, t_scatter) << ","
             << us(t_scatter, t_add) << "," << us(t_add, t_gather) << ","
             << us(t_gather, t_reduce) << "," << us(t_start, t_gather) << endl;

        free(v1); free(v2); free(v3);
    }

    free(l1); free(l2); free(l3);
    MPI_Finalize();
    return 0;
}