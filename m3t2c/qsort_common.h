// SIT315 M3.T2C - pieces shared by the MPI and the MPI + OpenCL sort.
//
// The dataset generator, checksum, sorted check, timer, argument parser and the
// local quicksort are lifted unchanged from m2t2c so the two programs here sort
// exactly the same bytes as the sequential, std::thread and OpenMP versions from
// module 2, and print a comparable checksum. m2t2c is submitted work and is not
// edited, which is why this is a copy rather than a shared include.

#ifndef QSORT_COMMON_H
#define QSORT_COMMON_H

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

namespace qs {

// Subarrays this size or smaller go to insertion sort instead of recursing.
const int64_t INSERTION_CUTOFF = 32;

enum class Pivot { Median3, Middle, Last, First };

inline Pivot parse_pivot(const std::string& s) {
    if (s == "median3") return Pivot::Median3;
    if (s == "middle")  return Pivot::Middle;
    if (s == "last")    return Pivot::Last;
    if (s == "first")   return Pivot::First;
    std::fprintf(stderr, "unknown pivot '%s' (median3|middle|last|first)\n", s.c_str());
    std::exit(1);
}

inline const char* pivot_name(Pivot p) {
    switch (p) {
        case Pivot::Median3: return "median3";
        case Pivot::Middle:  return "middle";
        case Pivot::Last:    return "last";
        case Pivot::First:   return "first";
    }
    return "?";
}

inline void insertion_sort(int32_t* a, int64_t lo, int64_t hi) {
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

inline int64_t choose_pivot_index(const int32_t* a, int64_t lo, int64_t hi, Pivot rule) {
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

// Hoare partition with the chosen pivot swapped to a[lo] first, so the pivot
// rule can change at run time without the loop spinning.
inline int64_t partition_hoare(int32_t* a, int64_t lo, int64_t hi, Pivot rule) {
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

// Recurse into the smaller half, loop on the larger one, so stack depth stays
// O(log n) even when the pivot rule splits badly.
inline void quicksort(int32_t* a, int64_t lo, int64_t hi, Pivot rule) {
    while (lo < hi) {
        if (hi - lo + 1 <= INSERTION_CUTOFF) {
            insertion_sort(a, lo, hi);
            return;
        }
        int64_t p = partition_hoare(a, lo, hi, rule);
        if (p - lo < hi - (p + 1)) {
            quicksort(a, lo, p, rule);
            lo = p + 1;
        } else {
            quicksort(a, p + 1, hi, rule);
            hi = p;
        }
    }
}

inline std::vector<int32_t> make_dataset(int64_t n, uint64_t seed, const std::string& pattern) {
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

// Sum of every element as unsigned, so it wraps instead of overflowing.
// Sorting is a permutation, so this survives the sort untouched. It catches a
// distributed bug that drops or duplicates an element, which an is-it-sorted
// check on its own would walk straight past.
inline uint64_t checksum(const int32_t* v, size_t n) {
    uint64_t s = 0;
    for (size_t i = 0; i < n; ++i) s += static_cast<uint32_t>(v[i]);
    return s;
}

inline bool is_sorted_asc(const int32_t* v, size_t n) {
    for (size_t i = 1; i < n; ++i)
        if (v[i - 1] > v[i]) return false;
    return true;
}

class Timer {
    std::chrono::steady_clock::time_point t0_;
public:
    void start() { t0_ = std::chrono::steady_clock::now(); }
    double stop_ms() const {
        auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0_).count();
    }
};

inline double median_of(std::vector<double> xs) {
    std::sort(xs.begin(), xs.end());
    size_t n = xs.size();
    if (n == 0) return 0.0;
    return (n % 2) ? xs[n / 2] : 0.5 * (xs[n / 2 - 1] + xs[n / 2]);
}

inline std::string arg_str(int argc, char** argv, const std::string& key, const std::string& fb) {
    for (int i = 1; i < argc - 1; ++i)
        if (key == argv[i]) return argv[i + 1];
    return fb;
}

inline long long arg_ll(int argc, char** argv, const std::string& key, long long fb) {
    for (int i = 1; i < argc - 1; ++i)
        if (key == argv[i]) return std::atoll(argv[i + 1]);
    return fb;
}

}  // namespace qs

#endif
