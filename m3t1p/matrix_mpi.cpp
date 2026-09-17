// SIT315 M3.T1P - Matrix multiplication distributed with MPI.
//
// Rank 0 generates A and B, broadcasts B whole, scatters A by row bands, and
// gathers the finished bands of C back. Each rank multiplies its own band with
// the same triple loop as m2t1p/matrix_seq.cpp, so the only difference from the
// sequential baseline is where the rows are computed.
//
// Build:  MPICH_CXX=g++ mpic++ -O2 -o matrix_mpi matrix_mpi.cpp
// Run:    mpirun -np 4 ./matrix_mpi 1000

#include "matmul_common.h"

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, procs, nameLen;
    char host[MPI_MAX_PROCESSOR_NAME];
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &procs);
    MPI_Get_processor_name(host, &nameLen);

    int N = (argc > 1) ? std::atoi(argv[1]) : 1000;
    if (N <= 0 || procs > N) {
        if (rank == 0)
            std::fprintf(stderr, "need N > 0 and at most one process per row (N=%d, procs=%d)\n",
                         N, procs);
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) matmul::announce("mpi", N, procs, 1);

    matmul::Bands b(N, procs, rank);
    std::vector<int> A, B, localA;
    std::vector<long long> C, localC(static_cast<size_t>(b.myRows) * N, 0);

    double t0 = 0, tGen = 0, tDist = 0, tComp = 0, tGath = 0;

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

    if (rank == 0) matmul::step("multiplying");

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

    std::printf("rank %d on %s computed rows %d..%d (%d rows)\n",
                rank, host, b.rowDispls[rank], b.rowDispls[rank] + b.myRows - 1, b.myRows);
    std::fflush(stdout);

    if (rank == 0) {
        tDist = tB - tA;
        tComp = tC - tB;
        tGath = tD - tC;
        matmul::report("mpi", N, procs, 1, tGen - t0, tDist, tComp, tGath,
                       matmul::checksum(C), matmul::verify(N, A, B, C));
    }

    MPI_Finalize();
    return 0;
}
