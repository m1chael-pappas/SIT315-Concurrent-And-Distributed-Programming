// SIT315 M3.T1P - Hybrid MPI and OpenMP matrix multiplication.
//
// The MPI layer is identical to matrix_mpi.cpp: B broadcast whole, A scattered
// by row bands, C gathered back. The difference is inside a rank, where the
// band is split again across OpenMP threads instead of being walked serially.
// Two levels of decomposition, MPI between processes and OpenMP within one.
//
// Build:  MPICH_CXX=g++ mpic++ -O2 -fopenmp -o matrix_mpi_omp matrix_mpi_omp.cpp
// Run:    mpirun -np 4 ./matrix_mpi_omp 1000 8

#include "matmul_common.h"
#include <omp.h>

int main(int argc, char* argv[]) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank, procs, nameLen;
    char host[MPI_MAX_PROCESSOR_NAME];
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &procs);
    MPI_Get_processor_name(host, &nameLen);

    int N = (argc > 1) ? std::atoi(argv[1]) : 1000;
    int T = (argc > 2) ? std::atoi(argv[2]) : 4;
    if (N <= 0 || T <= 0 || procs > N) {
        if (rank == 0)
            std::fprintf(stderr, "need N > 0, threads > 0 and at most one process per row "
                                 "(N=%d, threads=%d, procs=%d)\n", N, T, procs);
        MPI_Finalize();
        return 1;
    }
    omp_set_num_threads(T);

    if (rank == 0) matmul::announce("mpi+omp", N, procs, T);

    matmul::Bands b(N, procs, rank);
    std::vector<int> A, B, localA;
    std::vector<long long> C, localC(static_cast<size_t>(b.myRows) * N, 0);

    double t0 = 0, tGen = 0;

    if (rank == 0) {
        matmul::step("generating inputs");
        t0 = MPI_Wtime();
        matmul::generate(N, A, B);
        C.assign(static_cast<size_t>(N) * N, 0);
        tGen = MPI_Wtime();
    }

    if (rank == 0) matmul::step("broadcasting B and scattering A");
    MPI_Barrier(MPI_COMM_WORLD);
    double tA = MPI_Wtime();
    matmul::distribute(b, A, B, localA);
    MPI_Barrier(MPI_COMM_WORLD);
    double tB = MPI_Wtime();

    if (rank == 0) matmul::step("multiplying with OpenMP threads");

    // Only the communication happens outside the parallel region, so
    // MPI_THREAD_FUNNELED is enough: every MPI call is made by the main thread.
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < b.myRows; ++i) {
        for (int j = 0; j < N; ++j) {
            long long sum = 0;
            for (int k = 0; k < N; ++k) {
                sum += static_cast<long long>(localA[i * N + k]) * B[k * N + j];
            }
            localC[i * N + j] = sum;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double tC = MPI_Wtime();

    if (rank == 0) matmul::step("gathering C");
    matmul::collect(b, localC, C);
    MPI_Barrier(MPI_COMM_WORLD);
    double tD = MPI_Wtime();

    std::printf("rank %d on %s computed rows %d..%d (%d rows) with %d threads, MPI thread level %d\n",
                rank, host, b.rowDispls[rank], b.rowDispls[rank] + b.myRows - 1, b.myRows,
                T, provided);
    std::fflush(stdout);

    if (rank == 0) {
        matmul::report("mpi+omp", N, procs, T, tGen - t0, tB - tA, tC - tB, tD - tC,
                       matmul::checksum(C), matmul::verify(N, A, B, C));
    }

    MPI_Finalize();
    return 0;
}
