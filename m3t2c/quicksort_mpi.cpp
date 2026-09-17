// ============================================================================
// SIT315 Module 3, Task M3.T2C - Complex distributed computing
// Distributed quicksort with MPI, using parallel sorting by regular sampling.
// This is the baseline the MPI + OpenCL version is measured against.
//
// Each rank sorts its own chunk with the same quicksort as m2t2c, so the only
// thing that changes between the two programs here is where that local sort
// runs. The MPI layer lives in psrs.h and is shared.
//
// Build:  MPICH_CXX=g++ mpic++ -O2 -std=c++17 -Wall -Wextra -o quicksort_mpi quicksort_mpi.cpp
// Run:    mpirun -np 4 ./quicksort_mpi --n 16000000 --runs 5
// ============================================================================

#include "psrs.h"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, procs, nameLen;
    char host[MPI_MAX_PROCESSOR_NAME];
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &procs);
    MPI_Get_processor_name(host, &nameLen);

    const int64_t n       = qs::arg_ll(argc, argv, "--n", 16000000);
    const uint64_t seed   = static_cast<uint64_t>(qs::arg_ll(argc, argv, "--seed", 315));
    const int runs        = static_cast<int>(qs::arg_ll(argc, argv, "--runs", 5));
    const std::string pat = qs::arg_str(argc, argv, "--pattern", "random");
    const qs::Pivot rule  = qs::parse_pivot(qs::arg_str(argc, argv, "--pivot", "median3"));

    if (n <= 0 || runs <= 0) {
        if (rank == 0) std::fprintf(stderr, "--n and --runs must be positive\n");
        MPI_Finalize();
        return 1;
    }
    if (!psrs::usable(n, procs)) {
        if (rank == 0)
            std::fprintf(stderr, "need at least p*p elements to sample from: n=%lld, p=%d\n",
                         static_cast<long long>(n), procs);
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        char lib[MPI_MAX_LIBRARY_VERSION_STRING] = {0};
        int len = 0;
        MPI_Get_library_version(lib, &len);
        for (int i = 0; i < len; ++i) if (lib[i] == '\n') { lib[i] = '\0'; break; }

        std::fprintf(stderr, "[mpi] n=%lld procs=%d pivot=%s runs=%d on %s\n",
                     static_cast<long long>(n), procs, qs::pivot_name(rule), runs, host);
        std::fprintf(stderr, "      device: CPU only, no OpenCL in this build\n");
        std::fprintf(stderr, "      linked against %s\n", lib);
        std::fflush(stderr);
    }

    std::vector<int32_t> original, work;
    uint64_t want = 0;
    if (rank == 0) {
        std::fprintf(stderr, "  generating %lld elements (pattern=%s seed=%llu)\n",
                     static_cast<long long>(n), pat.c_str(),
                     static_cast<unsigned long long>(seed));
        std::fflush(stderr);
        original = qs::make_dataset(n, seed, pat);
        want = qs::checksum(original.data(), original.size());
        work.resize(original.size());
        std::printf("program   quicksort_mpi\n");
        std::printf("elements  %lld\n", static_cast<long long>(n));
        std::printf("bytes     %lld\n", static_cast<long long>(n) * 4);
        std::printf("processes %d\n", procs);
        std::printf("pivot     %s\n", qs::pivot_name(rule));
        std::printf("checksum  %llu\n", static_cast<unsigned long long>(want));
        std::printf("runs      %d\n\n", runs);
    }

    auto localSort = [rule](int32_t* a, int64_t count) {
        if (count > 1) qs::quicksort(a, 0, count - 1, rule);
    };

    std::vector<double> times;
    psrs::Phases last;
    for (int r = 0; r < runs; ++r) {
        if (rank == 0) {
            std::copy(original.begin(), original.end(), work.begin());
            std::fprintf(stderr, "  run %d of %d\n", r + 1, runs);
            std::fflush(stderr);
        }
        MPI_Barrier(MPI_COMM_WORLD);

        psrs::Phases ph;
        psrs::sort(work, n, rank, procs, localSort, ph);

        if (rank == 0) {
            bool ok = qs::is_sorted_asc(work.data(), work.size()) &&
                      qs::checksum(work.data(), work.size()) == want;
            std::printf("run %d    %10.2f ms   %s\n", r + 1, ph.total(), ok ? "verified" : "FAILED");
            if (!ok) { MPI_Abort(MPI_COMM_WORLD, 1); }
            times.push_back(ph.total());
            last = ph;
        }
    }

    if (rank == 0) {
        double med = qs::median_of(times);
        std::printf("\nmedian    %10.2f ms\n", med);
        std::printf("phases    scatter %.2f  localsort %.2f  pivots %.2f  "
                    "exchange %.2f  merge %.2f  gather %.2f  (last run, ms)\n",
                    last.scatter, last.localsort, last.pivots,
                    last.exchange, last.merge, last.gather);
        // The trailing device field keeps this line the same shape as the one
        // quicksort_mpi_ocl prints, so run_experiments.sh parses both the same.
        std::printf("CSV,mpi,%lld,%d,%s,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%llu,cpu native\n",
                    static_cast<long long>(n), procs, qs::pivot_name(rule), med,
                    last.scatter, last.localsort, last.pivots,
                    last.exchange, last.merge, last.gather,
                    static_cast<unsigned long long>(want));
    }

    MPI_Finalize();
    return 0;
}
