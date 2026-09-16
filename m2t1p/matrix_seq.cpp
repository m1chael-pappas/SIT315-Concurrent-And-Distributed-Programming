#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

static const unsigned SEED = 42; 

int main(int argc, char* argv[]) {
    int N = (argc > 1) ? std::atoi(argv[1]) : 1000;
    if (N <= 0) { std::fprintf(stderr, "N must be positive\n"); return 1; }

    // Flat 1D arrays
    std::vector<int> A(static_cast<size_t>(N) * N);
    std::vector<int> B(static_cast<size_t>(N) * N);
    std::vector<long long> C(static_cast<size_t>(N) * N, 0);

    // Initialisation (not timed)
    std::srand(SEED);
    for (size_t k = 0; k < A.size(); ++k) A[k] = std::rand() % 100;
    for (size_t k = 0; k < B.size(); ++k) B[k] = std::rand() % 100;

    // Multiplication (timed)
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            long long sum = 0;
            for (int k = 0; k < N; ++k) {
                sum += static_cast<long long>(A[i * N + k]) * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double seconds = std::chrono::duration<double>(t1 - t0).count();

    // Checksum for correctness comparison against the parallel versions
    unsigned long long checksum = 0;
    for (size_t k = 0; k < C.size(); ++k) checksum += static_cast<unsigned long long>(C[k]);

    // Write result matrix to file (not timed)
    std::FILE* f = std::fopen("output_seq.txt", "w");
    if (!f) { std::fprintf(stderr, "could not open output file\n"); return 1; }
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            std::fprintf(f, "%lld%c", C[i * N + j], (j == N - 1) ? '\n' : ' ');
        }
    }
    std::fclose(f);

    std::printf("version=sequential N=%d threads=1 time=%.6f s checksum=%llu\n",
                N, seconds, checksum);
    return 0;
}