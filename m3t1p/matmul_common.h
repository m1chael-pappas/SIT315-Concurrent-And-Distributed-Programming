// SIT315 M3.T1P - shared pieces of the three MPI matrix multiplication programs.
//
// The three programs differ only in how a rank multiplies its own band of rows.
// The row split, the input data, the communication pattern, the correctness
// check and the output format all live here, so the three produce numbers that
// can be compared directly against each other and against m2t1p/matrix_seq.cpp.

#ifndef MATMUL_COMMON_H
#define MATMUL_COMMON_H

#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace matmul {

// Matches m2t1p/matrix_seq.cpp so the checksums are comparable across tasks.
const unsigned SEED = 42;

// Above this N the sequential re-computation costs more than the run itself,
// so correctness falls back to comparing checksums between versions.
const int VERIFY_MAX = 1000;

// The row decomposition. Rank r owns rows [rowDispls[r], rowDispls[r]+rowCounts[r]).
// The elem* vectors are the same split measured in matrix elements, which is
// what MPI_Scatterv and MPI_Gatherv want.
struct Bands {
    int N = 0, procs = 0, rank = 0, myRows = 0;
    std::vector<int> rowCounts, rowDispls, elemCounts, elemDispls;

    Bands(int n, int p, int r) : N(n), procs(p), rank(r) {
        rowCounts.assign(p, 0);
        rowDispls.assign(p, 0);
        elemCounts.assign(p, 0);
        elemDispls.assign(p, 0);
        int base = N / p, extra = N % p, row = 0;
        for (int i = 0; i < p; ++i) {
            rowCounts[i] = base + (i < extra ? 1 : 0);
            rowDispls[i] = row;
            elemCounts[i] = rowCounts[i] * N;
            elemDispls[i] = row * N;
            row += rowCounts[i];
        }
        myRows = rowCounts[rank];
    }
};

// Rank 0 writes a startup banner and a line per phase to stderr, so a run is
// never a silent terminal. stderr rather than stdout keeps the CSV line clean
// for run_experiments.sh, which discards stderr.
//
// The banner also names the MPI library actually linked in. If a binary built
// against one MPI is launched by another's mpirun, the ranks die before this
// prints, so an empty terminal is itself the diagnosis.
inline void announce(const char* version, int N, int procs, int threads) {
    char lib[MPI_MAX_LIBRARY_VERSION_STRING] = {0};
    int len = 0;
    MPI_Get_library_version(lib, &len);
    for (int i = 0; i < len; ++i) {
        if (lib[i] == '\n') { lib[i] = '\0'; break; }
    }
    std::fprintf(stderr, "[%s] N=%d procs=%d threads=%d linked against %s\n",
                 version, N, procs, threads, lib);
    std::fflush(stderr);
}

inline void step(const char* what) {
    std::fprintf(stderr, "  %s\n", what);
    std::fflush(stderr);
}

inline void generate(int N, std::vector<int>& A, std::vector<int>& B) {
    A.resize(static_cast<size_t>(N) * N);
    B.resize(static_cast<size_t>(N) * N);
    std::srand(SEED);
    for (size_t k = 0; k < A.size(); ++k) A[k] = std::rand() % 100;
    for (size_t k = 0; k < B.size(); ++k) B[k] = std::rand() % 100;
}

// B goes to every rank whole because each rank touches every column of it.
// A is cut into row bands, so no element of C is computed twice.
inline void distribute(const Bands& b, std::vector<int>& A, std::vector<int>& B,
                       std::vector<int>& localA) {
    localA.resize(static_cast<size_t>(b.myRows) * b.N);
    B.resize(static_cast<size_t>(b.N) * b.N);
    MPI_Bcast(B.data(), b.N * b.N, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(b.rank == 0 ? A.data() : nullptr, b.elemCounts.data(), b.elemDispls.data(),
                 MPI_INT, localA.data(), b.myRows * b.N, MPI_INT, 0, MPI_COMM_WORLD);
}

inline void collect(const Bands& b, std::vector<long long>& localC, std::vector<long long>& C) {
    MPI_Gatherv(localC.data(), b.myRows * b.N, MPI_LONG_LONG,
                b.rank == 0 ? C.data() : nullptr, b.elemCounts.data(), b.elemDispls.data(),
                MPI_LONG_LONG, 0, MPI_COMM_WORLD);
}

inline unsigned long long checksum(const std::vector<long long>& C) {
    unsigned long long sum = 0;
    for (size_t k = 0; k < C.size(); ++k) sum += static_cast<unsigned long long>(C[k]);
    return sum;
}

// Recomputes the product sequentially on rank 0 and compares element by element.
inline const char* verify(int N, const std::vector<int>& A, const std::vector<int>& B,
                          const std::vector<long long>& C) {
    if (N > VERIFY_MAX) return "checksum-only";
    step("verifying against a sequential recompute");
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            long long sum = 0;
            for (int k = 0; k < N; ++k) {
                sum += static_cast<long long>(A[i * N + k]) * B[k * N + j];
            }
            if (sum != C[i * N + j]) return "FAILED";
        }
    }
    return "ok";
}

// One human line, one phase breakdown, one machine-readable line for the
// benchmark script. Times arrive in seconds and are printed in microseconds.
inline void report(const char* version, int N, int procs, int threads,
                   double gen, double dist, double comp, double gath,
                   unsigned long long sum, const char* verdict) {
    auto us = [](double s) { return static_cast<long>(s * 1e6); };
    double seconds = dist + comp + gath;
    std::printf("version=%s N=%d procs=%d threads=%d time=%.6f s checksum=%llu verify=%s\n",
                version, N, procs, threads, seconds, sum, verdict);
    std::printf("generate: %ld us  distribute: %ld us  compute: %ld us  gather: %ld us\n",
                us(gen), us(dist), us(comp), us(gath));
    std::printf("CSV,%s,%d,%d,%d,%ld,%ld,%ld,%ld,%.6f,%llu,%s\n",
                version, N, procs, threads, us(gen), us(dist), us(comp), us(gath),
                seconds, sum, verdict);
}

}  // namespace matmul

#endif
