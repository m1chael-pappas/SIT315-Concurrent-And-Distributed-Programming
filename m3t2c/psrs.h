// SIT315 M3.T2C - parallel sorting by regular sampling, the MPI layer.
//
// Both programs in this folder use this file for everything except the local
// sort of a rank's own chunk, which is passed in as a callable. quicksort_mpi
// passes the m2t2c quicksort, quicksort_mpi_ocl passes a bitonic sort that runs
// on an OpenCL device. Nothing else differs between them.
//
// The algorithm, in one pass:
//
//   1. rank 0 scatters the array in equal contiguous chunks
//   2. every rank sorts its own chunk                          <- the callable
//   3. every rank takes p evenly spaced samples of its sorted chunk
//   4. rank 0 gathers all p*p samples, sorts them, picks p-1 pivots, broadcasts
//   5. every rank splits its sorted chunk at those pivots into p buckets
//   6. all-to-all: bucket i from every rank goes to rank i
//   7. every rank merges the p sorted runs it received
//   8. rank 0 gathers the merged runs back in rank order
//
// After step 6 every element on rank r is <= every element on rank r+1, so the
// concatenation in rank order at step 8 is the fully sorted array. That is what
// makes the final gather a plain copy rather than another merge.

#ifndef PSRS_H
#define PSRS_H

#include <mpi.h>
#include "qsort_common.h"

namespace psrs {

struct Phases {
    double scatter = 0, localsort = 0, pivots = 0, exchange = 0, merge = 0, gather = 0;
    double total() const { return scatter + localsort + pivots + exchange + merge + gather; }
};

// Splits n elements across p ranks, giving the first n%p ranks one extra.
inline void split(int64_t n, int p, std::vector<int>& counts, std::vector<int>& displs) {
    counts.assign(p, 0);
    displs.assign(p, 0);
    int64_t base = n / p, extra = n % p, at = 0;
    for (int r = 0; r < p; ++r) {
        counts[r] = static_cast<int>(base + (r < extra ? 1 : 0));
        displs[r] = static_cast<int>(at);
        at += counts[r];
    }
}

// Sampling needs at least p elements per rank to take p samples from, so n must
// reach p*p before the decomposition makes sense at all.
inline bool usable(int64_t n, int p) { return p == 1 || n >= static_cast<int64_t>(p) * p; }

// Sorts `input` (valid on rank 0 only) and leaves the result there.
// LocalSort is called as localSort(int32_t* data, int64_t count).
template <typename LocalSort>
void sort(std::vector<int32_t>& input, int64_t n, int rank, int procs,
          LocalSort localSort, Phases& ph) {
    std::vector<int> counts, displs;
    split(n, procs, counts, displs);
    const int myCount = counts[rank];

    std::vector<int32_t> local(static_cast<size_t>(myCount));

    double t0 = MPI_Wtime();
    MPI_Scatterv(rank == 0 ? input.data() : nullptr, counts.data(), displs.data(), MPI_INT,
                 local.data(), myCount, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    localSort(local.data(), myCount);
    MPI_Barrier(MPI_COMM_WORLD);
    double t2 = MPI_Wtime();

    std::vector<int32_t> pivots(procs > 1 ? procs - 1 : 0);
    if (procs > 1) {
        // Regular sampling: p evenly spaced values from this rank's sorted chunk.
        std::vector<int32_t> mine(procs);
        const int64_t stride = myCount / procs;
        for (int i = 0; i < procs; ++i) mine[i] = local[static_cast<size_t>(i * stride)];

        std::vector<int32_t> all(static_cast<size_t>(procs) * procs);
        MPI_Gather(mine.data(), procs, MPI_INT, all.data(), procs, MPI_INT, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            std::sort(all.begin(), all.end());
            for (int i = 1; i < procs; ++i) pivots[i - 1] = all[static_cast<size_t>(i) * procs];
        }
        MPI_Bcast(pivots.data(), procs - 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double t3 = MPI_Wtime();

    // Bucket boundaries by binary search, because the chunk is already sorted.
    std::vector<int> sendCounts(procs), sendDispls(procs), recvCounts(procs), recvDispls(procs);
    {
        int64_t at = 0;
        for (int r = 0; r < procs; ++r) {
            int64_t end = myCount;
            if (r + 1 < procs) {
                end = std::upper_bound(local.begin(), local.end(), pivots[r]) - local.begin();
            }
            if (end < at) end = at;
            sendDispls[r] = static_cast<int>(at);
            sendCounts[r] = static_cast<int>(end - at);
            at = end;
        }
    }

    MPI_Alltoall(sendCounts.data(), 1, MPI_INT, recvCounts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    int64_t recvTotal = 0;
    for (int r = 0; r < procs; ++r) {
        recvDispls[r] = static_cast<int>(recvTotal);
        recvTotal += recvCounts[r];
    }
    std::vector<int32_t> received(static_cast<size_t>(recvTotal));
    MPI_Alltoallv(local.data(), sendCounts.data(), sendDispls.data(), MPI_INT,
                  received.data(), recvCounts.data(), recvDispls.data(), MPI_INT,
                  MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    double t4 = MPI_Wtime();

    // p sorted runs arrived back to back. Fold them together one at a time.
    {
        int64_t merged = recvCounts[0];
        for (int r = 1; r < procs; ++r) {
            std::inplace_merge(received.begin(),
                               received.begin() + merged,
                               received.begin() + merged + recvCounts[r]);
            merged += recvCounts[r];
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double t5 = MPI_Wtime();

    // Every rank's share is now a contiguous slice of the final order, but the
    // slices are uneven, so rank 0 needs the sizes before it can gather.
    int myFinal = static_cast<int>(recvTotal);
    std::vector<int> finalCounts(procs), finalDispls(procs);
    MPI_Gather(&myFinal, 1, MPI_INT, finalCounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        int64_t at = 0;
        for (int r = 0; r < procs; ++r) {
            finalDispls[r] = static_cast<int>(at);
            at += finalCounts[r];
        }
    }
    MPI_Gatherv(received.data(), myFinal, MPI_INT,
                rank == 0 ? input.data() : nullptr, finalCounts.data(), finalDispls.data(),
                MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    double t6 = MPI_Wtime();

    ph.scatter   = (t1 - t0) * 1000.0;
    ph.localsort = (t2 - t1) * 1000.0;
    ph.pivots    = (t3 - t2) * 1000.0;
    ph.exchange  = (t4 - t3) * 1000.0;
    ph.merge     = (t5 - t4) * 1000.0;
    ph.gather    = (t6 - t5) * 1000.0;
}

}  // namespace psrs

#endif
