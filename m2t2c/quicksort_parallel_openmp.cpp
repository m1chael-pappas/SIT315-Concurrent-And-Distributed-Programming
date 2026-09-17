// ============================================================================
// SIT315 Module 2, Task M2.T2C - Complex Threading
// Parallel quicksort using OpenMP tasks.
//
// The partition step stays sequential. What I parallelise is the recursion:
// after a partition the two halves cover disjoint index ranges, so two threads
// can sort them at the same time with no lock and no shared write.
//
// Build:  g++ -O2 -std=c++17 -Wall -Wextra -fopenmp quicksort_parallel_openmp.cpp -o quicksort_parallel_openmp
// Run:    ./quicksort_parallel_openmp --n 100000000 --threads 16 --pivot median3 --runs 5
//
// Michael Pappas
// ============================================================================


#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

// Subarrays this size or smaller go to insertion sort instead of recursing.
// 32 is the usual sweet spot for int32 on x86.
static const int64_t INSERTION_CUTOFF = 32;

enum class Pivot { Median3, Middle, Last, First };

static Pivot parse_pivot(const std::string& s) {
    if (s == "median3") return Pivot::Median3;
    if (s == "middle")  return Pivot::Middle;
    if (s == "last")    return Pivot::Last;
    if (s == "first")   return Pivot::First;
    std::fprintf(stderr, "unknown pivot '%s' (median3|middle|last|first)\n", s.c_str());
    std::exit(1);
}

static const char* pivot_name(Pivot p) {
    switch (p) {
        case Pivot::Median3: return "median3";
        case Pivot::Middle:  return "middle";
        case Pivot::Last:    return "last";
        case Pivot::First:   return "first";
    }
    return "?";
}

static void insertion_sort(int32_t* a, int64_t lo, int64_t hi) {
    for (int64_t i = lo + 1; i <= hi; ++i) {
        int32_t key = a[i];
        int64_t j = i - 1;
        while (j >= lo && a[j] > key) {
            a[j + 1] = a[j];
            --j;
        }
        a[j + 1] = key;
    }
}

static int64_t choose_pivot_index(const int32_t* a, int64_t lo, int64_t hi, Pivot rule) {
    int64_t mid = lo + (hi - lo) / 2;
    switch (rule) {
        case Pivot::First:  return lo;
        case Pivot::Last:   return hi;
        case Pivot::Middle: return mid;
        case Pivot::Median3: {
            int32_t A = a[lo], B = a[mid], C = a[hi];
            if ((A <= B && B <= C) || (C <= B && B <= A)) return mid;
            if ((B <= A && A <= C) || (C <= A && A <= B)) return lo;
            return hi;
        }
    }
    return mid;
}

// Hoare partition. Returns j, so the halves are [lo, j] and [j+1, hi].
// I swap the chosen pivot into a[lo] before the loop. Hoare's scheme spins
// forever when the pivot value sits at a[hi], so moving it to the front lets me
// switch pivot rules at runtime without touching the loop itself.
static int64_t partition_hoare(int32_t* a, int64_t lo, int64_t hi, Pivot rule) {
    int64_t pi = choose_pivot_index(a, lo, hi, rule);
    std::swap(a[lo], a[pi]);
    int32_t p = a[lo];

    int64_t i = lo - 1;
    int64_t j = hi + 1;
    for (;;) {
        do { ++i; } while (a[i] < p);
        do { --j; } while (a[j] > p);
        if (i >= j) return j;
        std::swap(a[i], a[j]);
    }
}

// ---------------------------------------------------------------- dataset ---

// Builds the input in memory. mt19937_64 with a fixed seed gives the same
// values on every machine and every run, so the sequential and parallel
// programs sort identical data without needing to share a file. The printed
// checksum is what proves they matched.
static std::vector<int32_t> make_dataset(int64_t n, uint64_t seed, const std::string& pattern) {
    std::vector<int32_t> v(static_cast<size_t>(n));
    if (pattern == "sorted") {
        for (int64_t i = 0; i < n; ++i) v[i] = static_cast<int32_t>(i);
    } else if (pattern == "reverse") {
        for (int64_t i = 0; i < n; ++i) v[i] = static_cast<int32_t>(n - i);
    } else {
        std::mt19937_64 rng(seed);
        std::uniform_int_distribution<int32_t> dist(-1000000000, 1000000000);
        for (int64_t i = 0; i < n; ++i) v[i] = dist(rng);
    }
    return v;
}

static std::vector<int32_t> load_binary(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        std::fprintf(stderr, "cannot open input: %s\n", path.c_str());
        std::exit(1);
    }
    std::streamsize bytes = f.tellg();
    f.seekg(0);
    std::vector<int32_t> v(static_cast<size_t>(bytes / 4));
    f.read(reinterpret_cast<char*>(v.data()), bytes);
    return v;
}

static void save_binary(const std::string& path, const std::vector<int32_t>& v) {
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "cannot write: %s\n", path.c_str());
        std::exit(1);
    }
    f.write(reinterpret_cast<const char*>(v.data()),
            static_cast<std::streamsize>(v.size() * 4));
    std::printf("saved     %s (%zu bytes)\n", path.c_str(), v.size() * 4);
}

// ------------------------------------------------------------ correctness ---

// Sum of every element, taken as unsigned so it wraps instead of overflowing.
// Sorting is a permutation, so this value has to survive the sort untouched. It
// catches a parallel bug that drops or duplicates an element, which a plain
// is-it-sorted check would walk straight past.
static uint64_t checksum(const std::vector<int32_t>& v) {
    uint64_t s = 0;
    for (int32_t x : v) s += static_cast<uint32_t>(x);
    return s;
}

static bool is_sorted_asc(const std::vector<int32_t>& v) {
    for (size_t i = 1; i < v.size(); ++i)
        if (v[i - 1] > v[i]) return false;
    return true;
}

// ----------------------------------------------------------------- timing ---

class Timer {
    std::chrono::steady_clock::time_point t0_;
public:
    void start() { t0_ = std::chrono::steady_clock::now(); }
    double stop_ms() const {
        auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0_).count();
    }
};

static double median_of(std::vector<double> xs) {
    std::sort(xs.begin(), xs.end());
    size_t n = xs.size();
    if (n == 0) return 0.0;
    return (n % 2) ? xs[n / 2] : 0.5 * (xs[n / 2 - 1] + xs[n / 2]);
}

// ------------------------------------------------------------------- args ---

static std::string arg_str(int argc, char** argv, const std::string& key, const std::string& fb) {
    for (int i = 1; i < argc - 1; ++i)
        if (key == argv[i]) return argv[i + 1];
    return fb;
}

static long long arg_ll(int argc, char** argv, const std::string& key, long long fb) {
    for (int i = 1; i < argc - 1; ++i)
        if (key == argv[i]) return std::atoll(argv[i + 1]);
    return fb;
}

// Loads the dataset from --input if given, otherwise builds it from --n,
// --seed and --pattern. --save writes whatever it ended up with to disk.
static std::vector<int32_t> get_input(int argc, char** argv) {
    std::string input   = arg_str(argc, argv, "--input", "");
    std::string save    = arg_str(argc, argv, "--save", "");
    std::string pattern = arg_str(argc, argv, "--pattern", "random");
    long long n         = arg_ll(argc, argv, "--n", 10000000);
    long long seed      = arg_ll(argc, argv, "--seed", 315);

    std::vector<int32_t> v;
    if (!input.empty()) {
        v = load_binary(input);
        std::printf("source    file %s\n", input.c_str());
    } else {
        v = make_dataset(n, static_cast<uint64_t>(seed), pattern);
        std::printf("source    generated (pattern=%s seed=%lld)\n", pattern.c_str(), seed);
    }
    if (!save.empty()) save_binary(save, v);
    return v;
}

// ========================================================================== //
// END SHARED SECTION
// ========================================================================== //

#include <omp.h>

// ========================== OPENMP TASK QUICKSORT ===========================

// Below task_cutoff elements I stop spawning tasks and finish the subarray on
// the current thread. Without that guard the program creates a task at every
// recursion level down to 32 elements, and the runtime spends longer managing
// tasks than it does sorting.
//
// Shared: the array, through the pointer. Private per task: lo, hi, p and the
// pivot value, all copied in by firstprivate. Two sibling tasks never touch the
// same index, which is why there is no mutex anywhere in this file.
static void quicksort(int32_t* a, int64_t lo, int64_t hi, Pivot rule, int64_t task_cutoff) {
    if (lo >= hi) return;
    if (hi - lo + 1 <= INSERTION_CUTOFF) {
        insertion_sort(a, lo, hi);
        return;
    }

    int64_t p = partition_hoare(a, lo, hi, rule);

    if (hi - lo + 1 > task_cutoff) {
        #pragma omp task default(none) firstprivate(a, lo, p, rule, task_cutoff)
        quicksort(a, lo, p, rule, task_cutoff);

        #pragma omp task default(none) firstprivate(a, p, hi, rule, task_cutoff)
        quicksort(a, p + 1, hi, rule, task_cutoff);

        // Both halves must finish before this call returns, otherwise the
        // parent would report a sorted array while children are still running.
        #pragma omp taskwait
    } else {
        quicksort(a, lo, p, rule, task_cutoff);
        quicksort(a, p + 1, hi, rule, task_cutoff);
    }
}

int main(int argc, char** argv) {
    Pivot rule        = parse_pivot(arg_str(argc, argv, "--pivot", "median3"));
    int runs          = static_cast<int>(arg_ll(argc, argv, "--runs", 5));
    int threads       = static_cast<int>(arg_ll(argc, argv, "--threads", 8));
    int64_t task_cutoff = arg_ll(argc, argv, "--task-cutoff", 100000);

    omp_set_num_threads(threads);

    std::printf("program      quicksort_parallel_openmp\n");
    std::vector<int32_t> original = get_input(argc, argv);
    uint64_t want = checksum(original);

    std::printf("elements     %zu\n", original.size());
    std::printf("bytes        %zu\n", original.size() * 4);
    std::printf("threads      %d\n", threads);
    std::printf("pivot        %s\n", pivot_name(rule));
    std::printf("task-cutoff  %lld\n", static_cast<long long>(task_cutoff));
    std::printf("checksum     %llu\n", static_cast<unsigned long long>(want));
    std::printf("runs         %d\n\n", runs);

    std::vector<double> times;
    for (int r = 0; r < runs; ++r) {
        std::vector<int32_t> a = original;
        int32_t* data = a.data();
        int64_t n = static_cast<int64_t>(a.size());

        Timer t;
        t.start();
        #pragma omp parallel default(none) shared(data, n, rule, task_cutoff)
        {
            // One thread walks the recursion and creates tasks. The rest sit in
            // the team and pick tasks up off the queue.
            #pragma omp single
            quicksort(data, 0, n - 1, rule, task_cutoff);
        }
        double ms = t.stop_ms();

        bool ok = is_sorted_asc(a) && checksum(a) == want;
        std::printf("run %d       %10.2f ms   %s\n", r + 1, ms, ok ? "verified" : "FAILED");
        if (!ok) return 1;
        times.push_back(ms);
    }

    double med = median_of(times);
    std::printf("\nmedian       %10.2f ms\n", med);
    std::printf("CSV,omp,%zu,%d,%s,%.3f\n", original.size(), threads, pivot_name(rule), med);
    return 0;
}