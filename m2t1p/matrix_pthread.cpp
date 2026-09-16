#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <pthread.h>
#include <vector>

static const unsigned SEED = 42;

struct ThreadArgs {
    const int* A;
    const int* B;
    long long* C;
    int N;
    int rowStart; // first row this thread computes (inclusive)
    int rowEnd;   // last row this thread computes (exclusive)
};

void* worker(void* p) {
    ThreadArgs* a = static_cast<ThreadArgs*>(p);
    const int N = a->N;
    for (int i = a->rowStart; i < a->rowEnd; ++i) {
        for (int j = 0; j < N; ++j) {
            long long sum = 0;
            for (int k = 0; k < N; ++k) {
                sum += static_cast<long long>(a->A[i * N + k]) * a->B[k * N + j];
            }
            a->C[i * N + j] = sum;
        }
    }
    return nullptr;
}

int main(int argc, char* argv[]) {
    int N = (argc > 1) ? std::atoi(argv[1]) : 1000;
    int T = (argc > 2) ? std::atoi(argv[2]) : 4;
    if (N <= 0 || T <= 0) { std::fprintf(stderr, "N and threads must be positive\n"); return 1; }
    if (T > N) T = N; 

    std::vector<int> A(static_cast<size_t>(N) * N);
    std::vector<int> B(static_cast<size_t>(N) * N);
    std::vector<long long> C(static_cast<size_t>(N) * N, 0);

    // Initialisation (not timed)
    std::srand(SEED);
    for (size_t k = 0; k < A.size(); ++k) A[k] = std::rand() % 100;
    for (size_t k = 0; k < B.size(); ++k) B[k] = std::rand() % 100;

    // Multiplication
    auto t0 = std::chrono::high_resolution_clock::now();

    std::vector<pthread_t> threads(T);
    std::vector<ThreadArgs> args(T);
    int rowsPerThread = N / T;
    int remainder = N % T;
    int row = 0;
    for (int t = 0; t < T; ++t) {
        int count = rowsPerThread + (t < remainder ? 1 : 0);
        args[t] = ThreadArgs{A.data(), B.data(), C.data(), N, row, row + count};
        row += count;
        pthread_create(&threads[t], nullptr, worker, &args[t]);
    }
    for (int t = 0; t < T; ++t) pthread_join(threads[t], nullptr);

    auto t1 = std::chrono::high_resolution_clock::now();
    double seconds = std::chrono::duration<double>(t1 - t0).count();

    unsigned long long checksum = 0;
    for (size_t k = 0; k < C.size(); ++k) checksum += static_cast<unsigned long long>(C[k]);

    std::FILE* f = std::fopen("output_pthread.txt", "w");
    if (!f) { std::fprintf(stderr, "could not open output file\n"); return 1; }
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            std::fprintf(f, "%lld%c", C[i * N + j], (j == N - 1) ? '\n' : ' ');
        }
    }
    std::fclose(f);

    std::printf("version=pthread N=%d threads=%d time=%.6f s checksum=%llu\n",
                N, T, seconds, checksum);
    return 0;
}