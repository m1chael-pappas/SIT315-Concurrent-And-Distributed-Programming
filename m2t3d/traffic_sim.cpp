/*
 * SIT315 Module 2 - Task M2.T3D - Traffic Control Simulator
 * Michael Pappas
 *
 * One binary, three modes:
 *
 *   gen  - writes a traffic data file, one line per measurement
 *          format: YYYY-MM-DD HH:MM,TLxxxx,cars
 *          every traffic light reports once every 5 minutes, so 12 rows per light per hour
 *
 *   seq  - single threaded baseline. Reads the file, totals cars per (hour, light),
 *          then ranks the top N lights for each hour.
 *
 *   par  - bounded buffer producer/consumer. Producers each own a byte range of the file,
 *          parse lines into fixed size blocks of records, and push blocks into a bounded
 *          queue. Consumers pop blocks and add the car counts into a summary structure.
 *          Ranking happens once at the end, after every record has been counted.
 *
 * Build:
 *   g++ -std=c++17 -O2 -pthread -Wall -o traffic_sim traffic_sim.cpp
 *
 * Examples:
 *   ./traffic_sim gen --out sample_input.csv --lights 20 --hours 4
 *   ./traffic_sim seq --in traffic_medium.csv --top 5 --print-hours 3
 *   ./traffic_sim par --in traffic_medium.csv --producers 4 --consumers 4 \
 *                     --buffer 64 --block 4096 --summary local
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace
{

// ---------------------------------------------------------------------------
// Basic types
// ---------------------------------------------------------------------------

// One parsed measurement. 12 bytes, trivially copyable, so a block of these
// moves through the queue as one contiguous allocation.
struct Record
{
    uint32_t hour;   // hour index relative to the first timestamp in the file
    uint32_t light;  // traffic light id
    uint32_t cars;   // cars counted in this 5 minute window
};

// The unit of work that travels through the bounded buffer.
using Block = std::vector<Record>;

using Clock = std::chrono::steady_clock;

double msSince(const Clock::time_point& start)
{
    const auto now = Clock::now();
    return std::chrono::duration<double, std::milli>(now - start).count();
}

// ---------------------------------------------------------------------------
// Calendar helpers (Howard Hinnant's civil date algorithms)
// ---------------------------------------------------------------------------

long long daysFromCivil(int y, unsigned m, unsigned d)
{
    y -= m <= 2;
    const int era = (y >= 0 ? y : y - 399) / 400;
    const unsigned yoe = static_cast<unsigned>(y - era * 400);
    const unsigned doy = (153u * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
    const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return static_cast<long long>(era) * 146097 + static_cast<long long>(doe) - 719468;
}

void civilFromDays(long long z, int& y, unsigned& m, unsigned& d)
{
    z += 719468;
    const long long era = (z >= 0 ? z : z - 146096) / 146097;
    const unsigned doe = static_cast<unsigned>(z - era * 146097);
    const unsigned yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    const long long yy = static_cast<long long>(yoe) + era * 400;
    const unsigned doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    const unsigned mp = (5 * doy + 2) / 153;
    d = doy - (153 * mp + 2) / 5 + 1;
    m = mp + (mp < 10 ? 3 : -9);
    y = static_cast<int>(yy + (m <= 2));
}

std::string formatHour(long long absHour)
{
    const long long day = absHour / 24;
    const int hourOfDay = static_cast<int>(absHour % 24);
    int y = 0;
    unsigned m = 0;
    unsigned d = 0;
    civilFromDays(day, y, m, d);

    char buf[32];
    std::snprintf(buf, sizeof(buf), "%04d-%02u-%02u %02d:00", y, m, d, hourOfDay);
    return std::string(buf);
}

// ---------------------------------------------------------------------------
// Line parsing
// ---------------------------------------------------------------------------

// Expected line: YYYY-MM-DD HH:MM,TL0042,37
// The timestamp is fixed width (16 chars) so the date fields come out of known
// offsets. The light id and car count are parsed digit by digit, which is a lot
// cheaper than sscanf or istringstream at tens of millions of lines.
bool parseLine(const char* s, size_t n, long long& absHour, uint32_t& light, uint32_t& cars)
{
    if (n < 20)
    {
        return false;
    }
    if (s[4] != '-' || s[7] != '-' || s[10] != ' ' || s[13] != ':' || s[16] != ',')
    {
        return false;
    }

    const int y = (s[0] - '0') * 1000 + (s[1] - '0') * 100 + (s[2] - '0') * 10 + (s[3] - '0');
    const unsigned mo = static_cast<unsigned>((s[5] - '0') * 10 + (s[6] - '0'));
    const unsigned da = static_cast<unsigned>((s[8] - '0') * 10 + (s[9] - '0'));
    const int hh = (s[11] - '0') * 10 + (s[12] - '0');

    if (mo < 1 || mo > 12 || da < 1 || da > 31 || hh < 0 || hh > 23)
    {
        return false;
    }

    size_t i = 17;
    while (i < n && (s[i] < '0' || s[i] > '9'))
    {
        ++i;  // skip the "TL" label
    }

    uint32_t id = 0;
    bool sawDigit = false;
    while (i < n && s[i] >= '0' && s[i] <= '9')
    {
        id = id * 10 + static_cast<uint32_t>(s[i] - '0');
        ++i;
        sawDigit = true;
    }
    if (!sawDigit || i >= n || s[i] != ',')
    {
        return false;
    }
    ++i;

    uint32_t c = 0;
    sawDigit = false;
    while (i < n && s[i] >= '0' && s[i] <= '9')
    {
        c = c * 10 + static_cast<uint32_t>(s[i] - '0');
        ++i;
        sawDigit = true;
    }
    if (!sawDigit)
    {
        return false;
    }

    absHour = daysFromCivil(y, mo, da) * 24 + hh;
    light = id;
    cars = c;
    return true;
}

// ---------------------------------------------------------------------------
// Bounded buffer
// ---------------------------------------------------------------------------
//
// A fixed capacity queue of blocks guarded by one mutex and two condition
// variables. Producers block on notFull_ when the queue is at capacity.
// Consumers block on notEmpty_ when the queue is drained. Both waits sit in a
// while loop so a spurious wakeup re-checks the predicate.
//
// Termination: main joins the producers, then calls close(). close() flips
// closed_ and wakes every waiting consumer. A consumer that finds the queue
// empty while closed_ is set returns false and exits its loop, so no consumer
// can be left parked on notEmpty_ forever.
//
// stop() is the abort path. A producer that hits a malformed line calls it,
// which wakes every blocked thread on both condition variables and makes push()
// and pop() return false immediately.
class BoundedBuffer
{
public:
    explicit BoundedBuffer(size_t capacityBlocks)
        : capacity_(capacityBlocks == 0 ? 1 : capacityBlocks)
    {
    }

    bool push(Block&& block)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        while (queue_.size() >= capacity_ && !stopped_)
        {
            ++pushWaits_;
            notFull_.wait(lock);
        }
        if (stopped_)
        {
            return false;
        }

        queue_.push(std::move(block));
        ++blocksPushed_;
        if (queue_.size() > maxDepth_)
        {
            maxDepth_ = queue_.size();
        }
        lock.unlock();

        notEmpty_.notify_one();
        return true;
    }

    bool pop(Block& out)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        while (queue_.empty() && !closed_ && !stopped_)
        {
            ++popWaits_;
            notEmpty_.wait(lock);
        }
        if (stopped_ || queue_.empty())
        {
            return false;
        }

        out = std::move(queue_.front());
        queue_.pop();
        ++blocksPopped_;
        lock.unlock();

        notFull_.notify_one();
        return true;
    }

    void close()
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            closed_ = true;
        }
        notEmpty_.notify_all();
    }

    void stop()
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopped_ = true;
        }
        notEmpty_.notify_all();
        notFull_.notify_all();
    }

    bool stopped() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return stopped_;
    }

    void stats(uint64_t& pushed, uint64_t& popped, uint64_t& pushWaits,
               uint64_t& popWaits, size_t& maxDepth) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        pushed = blocksPushed_;
        popped = blocksPopped_;
        pushWaits = pushWaits_;
        popWaits = popWaits_;
        maxDepth = maxDepth_;
    }

    size_t capacity() const
    {
        return capacity_;
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable notFull_;
    std::condition_variable notEmpty_;
    std::queue<Block> queue_;

    const size_t capacity_;
    bool closed_ = false;
    bool stopped_ = false;

    uint64_t blocksPushed_ = 0;
    uint64_t blocksPopped_ = 0;
    uint64_t pushWaits_ = 0;
    uint64_t popWaits_ = 0;
    size_t maxDepth_ = 0;
};

// ---------------------------------------------------------------------------
// Summary structure
// ---------------------------------------------------------------------------
//
// Totals live in a dense array indexed by hour * lights + light. Light ids are
// dense and hours are contiguous, so the array wastes nothing and every update
// is a single indexed add with no hashing and no allocation.
//
// Three update strategies, switchable at runtime so the report can compare them:
//   local  - every consumer owns a private array, main folds them together after
//            the joins. No synchronisation on the hot path.
//   atomic - one shared array of atomics, updated with relaxed fetch_add.
//            Non-blocking, but every update touches shared cache lines.
//   shared - one shared plain array behind a single mutex, locked once per
//            record. Correct, and deliberately the slow one.

enum class SummaryMode
{
    Local,
    Shared,
    Atomic
};

struct SummaryStore
{
    SummaryMode mode = SummaryMode::Local;
    uint32_t lights = 0;
    uint32_t hours = 0;

    std::vector<uint64_t> global;                       // final totals, and the live array in Shared mode
    std::mutex globalMutex;                             // Shared mode only
    std::unique_ptr<std::atomic<uint64_t>[]> atomics;   // Atomic mode only
    std::vector<std::vector<uint64_t>> locals;          // Local mode only, one per consumer

    size_t cells() const
    {
        return static_cast<size_t>(hours) * static_cast<size_t>(lights);
    }
};

inline void applyRecord(SummaryStore& store, int consumerId, const Record& r)
{
    const size_t idx = static_cast<size_t>(r.hour) * store.lights + r.light;

    switch (store.mode)
    {
        case SummaryMode::Local:
            store.locals[static_cast<size_t>(consumerId)][idx] += r.cars;
            break;

        case SummaryMode::Atomic:
            store.atomics[idx].fetch_add(r.cars, std::memory_order_relaxed);
            break;

        case SummaryMode::Shared:
        {
            std::lock_guard<std::mutex> lock(store.globalMutex);
            store.global[idx] += r.cars;
            break;
        }
    }
}

// Collapse whatever the consumers wrote into store.global.
//
// The local merge costs consumers * cells additions, so it grows with the
// consumer count and would eat the gain at high thread counts if it stayed
// serial. Cell ranges are disjoint, so the merge splits across threads with no
// locking at all: thread k owns [lo, hi) and is the only writer for that slice.
void finaliseSummary(SummaryStore& store, int mergeThreads)
{
    const size_t n = store.cells();
    if (mergeThreads < 1)
    {
        mergeThreads = 1;
    }

    auto runRanges = [&](const std::function<void(size_t, size_t)>& body)
    {
        if (mergeThreads == 1 || n < 65536)
        {
            body(0, n);
            return;
        }
        std::vector<std::thread> workers;
        workers.reserve(static_cast<size_t>(mergeThreads));
        for (int k = 0; k < mergeThreads; ++k)
        {
            const size_t lo = n * static_cast<size_t>(k) / static_cast<size_t>(mergeThreads);
            const size_t hi = n * static_cast<size_t>(k + 1) / static_cast<size_t>(mergeThreads);
            workers.emplace_back([lo, hi, &body]() { body(lo, hi); });
        }
        for (auto& w : workers)
        {
            w.join();
        }
    };

    if (store.mode == SummaryMode::Local)
    {
        store.global.assign(n, 0);
        runRanges([&](size_t lo, size_t hi)
        {
            for (const auto& local : store.locals)
            {
                for (size_t i = lo; i < hi; ++i)
                {
                    store.global[i] += local[i];
                }
            }
        });
    }
    else if (store.mode == SummaryMode::Atomic)
    {
        store.global.assign(n, 0);
        runRanges([&](size_t lo, size_t hi)
        {
            for (size_t i = lo; i < hi; ++i)
            {
                store.global[i] = store.atomics[i].load(std::memory_order_relaxed);
            }
        });
    }
    // Shared mode already wrote straight into store.global.
}

// FNV-1a over every non-zero total, walked in hour then light order. Two runs
// that agree on this value counted exactly the same cars into exactly the same
// buckets, which is how the sequential and parallel results get compared.
uint64_t summaryChecksum(const SummaryStore& store)
{
    uint64_t h = 1469598103934665603ull;

    auto mix = [&h](uint64_t value)
    {
        for (int b = 0; b < 8; ++b)
        {
            h ^= (value >> (b * 8)) & 0xFFull;
            h *= 1099511628211ull;
        }
    };

    for (uint32_t hour = 0; hour < store.hours; ++hour)
    {
        const size_t base = static_cast<size_t>(hour) * store.lights;
        for (uint32_t light = 0; light < store.lights; ++light)
        {
            const uint64_t total = store.global[base + light];
            if (total != 0)
            {
                mix(hour);
                mix(light);
                mix(total);
            }
        }
    }
    return h;
}

// ---------------------------------------------------------------------------
// Ranking
// ---------------------------------------------------------------------------

struct LightTotal
{
    uint32_t light;
    uint64_t cars;
};

bool busierFirst(const LightTotal& a, const LightTotal& b)
{
    if (a.cars != b.cars)
    {
        return a.cars > b.cars;
    }
    return a.light < b.light;  // stable tie break so both modes agree
}

// Ranks every hour. Prints the first printHours of them, and writes all of them
// to outPath when a path is given.
void rankAndReport(const SummaryStore& store, long long baseHour, int topN,
                   int printHours, const std::string& outPath, double& rankMs)
{
    const auto start = Clock::now();

    std::ofstream out;
    if (!outPath.empty())
    {
        out.open(outPath, std::ios::binary);
    }

    std::vector<LightTotal> scratch;
    scratch.reserve(store.lights);

    std::ostringstream console;
    int printed = 0;

    for (uint32_t hour = 0; hour < store.hours; ++hour)
    {
        scratch.clear();
        const size_t base = static_cast<size_t>(hour) * store.lights;
        for (uint32_t light = 0; light < store.lights; ++light)
        {
            const uint64_t total = store.global[base + light];
            if (total != 0)
            {
                scratch.push_back(LightTotal{light, total});
            }
        }
        if (scratch.empty())
        {
            continue;
        }

        const size_t keep = std::min(static_cast<size_t>(topN), scratch.size());
        std::partial_sort(scratch.begin(), scratch.begin() + static_cast<long>(keep),
                          scratch.end(), busierFirst);

        const std::string label = formatHour(baseHour + hour);

        if (printed < printHours)
        {
            console << "  " << label << "\n";
            for (size_t i = 0; i < keep; ++i)
            {
                console << "    " << std::setw(2) << (i + 1) << ". TL"
                        << std::setw(4) << std::setfill('0') << scratch[i].light
                        << std::setfill(' ') << "   " << std::setw(8) << scratch[i].cars
                        << " cars\n";
            }
            ++printed;
        }

        if (out.is_open())
        {
            out << label;
            for (size_t i = 0; i < keep; ++i)
            {
                out << ',' << scratch[i].light << ':' << scratch[i].cars;
            }
            out << '\n';
        }
    }

    rankMs = msSince(start);

    std::cout << "Top " << topN << " congested lights per hour (first "
              << printHours << " hours shown):\n";
    std::cout << console.str();
}

// ---------------------------------------------------------------------------
// File probing
// ---------------------------------------------------------------------------
//
// Reads the first and last line of the file. The first line gives the base hour.
// The generator writes hour major and light minor, so the last line gives the
// highest hour and the highest light id, which sizes the summary array without
// a full pre-pass over the data.

struct FileProbe
{
    long long baseHour = 0;
    long long lastHour = 0;
    uint32_t lastLight = 0;
    uint64_t size = 0;
};

bool probeFile(const std::string& path, FileProbe& probe, std::string& error)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
    {
        error = "cannot open input file: " + path;
        return false;
    }

    in.seekg(0, std::ios::end);
    const std::streamoff size = in.tellg();
    if (size <= 0)
    {
        error = "input file is empty: " + path;
        return false;
    }
    probe.size = static_cast<uint64_t>(size);

    in.seekg(0, std::ios::beg);
    std::string first;
    if (!std::getline(in, first))
    {
        error = "cannot read the first line of " + path;
        return false;
    }
    if (!first.empty() && first.back() == '\r')
    {
        first.pop_back();
    }

    uint32_t light = 0;
    uint32_t cars = 0;
    if (!parseLine(first.c_str(), first.size(), probe.baseHour, light, cars))
    {
        error = "first line is not in the expected format: " + first;
        return false;
    }

    const std::streamoff tailBytes = std::min<std::streamoff>(size, 4096);
    in.clear();
    in.seekg(size - tailBytes, std::ios::beg);
    std::vector<char> tail(static_cast<size_t>(tailBytes));
    in.read(tail.data(), tailBytes);

    size_t end = tail.size();
    while (end > 0 && (tail[end - 1] == '\n' || tail[end - 1] == '\r'))
    {
        --end;
    }
    size_t begin = end;
    while (begin > 0 && tail[begin - 1] != '\n')
    {
        --begin;
    }
    if (begin >= end)
    {
        error = "cannot read the last line of " + path;
        return false;
    }

    if (!parseLine(tail.data() + begin, end - begin, probe.lastHour, probe.lastLight, cars))
    {
        error = "last line is not in the expected format";
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

struct Options
{
    std::string mode;
    std::string inPath;
    std::string outPath = "traffic_data.csv";
    std::string outSummary;

    uint32_t lights = 0;   // 0 means take it from the file
    uint32_t hours = 0;    // 0 means take it from the file
    uint64_t seed = 42;

    int producers = 4;
    int consumers = 4;
    size_t bufferBlocks = 64;
    size_t blockRecords = 4096;
    SummaryMode summary = SummaryMode::Local;
    std::string summaryName = "local";

    int topN = 5;
    int printHours = 3;
    bool progress = false;
    bool csv = false;
};

void usage()
{
    std::cout <<
        "SIT315 M2.T3D traffic control simulator\n"
        "\n"
        "  traffic_sim gen --out FILE --lights N --hours H [--seed S]\n"
        "  traffic_sim seq --in FILE [--top N] [--print-hours K] [--out-summary FILE] [--csv]\n"
        "  traffic_sim par --in FILE [--producers P] [--consumers C] [--buffer B]\n"
        "                  [--block R] [--summary local|atomic|shared]\n"
        "                  [--top N] [--print-hours K] [--out-summary FILE]\n"
        "                  [--progress] [--csv]\n"
        "\n"
        "  --buffer  queue capacity in blocks (default 64)\n"
        "  --block   records per block (default 4096)\n"
        "  --summary how consumers accumulate totals (default local)\n"
        "  --lights/--hours override the dimensions probed from the file\n";
}

bool parseArgs(int argc, char** argv, Options& opt, std::string& error)
{
    if (argc < 2)
    {
        error = "no mode given";
        return false;
    }
    opt.mode = argv[1];
    if (opt.mode != "gen" && opt.mode != "seq" && opt.mode != "par")
    {
        error = "unknown mode: " + opt.mode;
        return false;
    }

    for (int i = 2; i < argc; ++i)
    {
        const std::string flag = argv[i];
        auto next = [&](std::string& dest) -> bool
        {
            if (i + 1 >= argc)
            {
                error = "missing value for " + flag;
                return false;
            }
            dest = argv[++i];
            return true;
        };

        std::string value;
        if (flag == "--in")
        {
            if (!next(value)) return false;
            opt.inPath = value;
        }
        else if (flag == "--out")
        {
            if (!next(value)) return false;
            opt.outPath = value;
        }
        else if (flag == "--out-summary")
        {
            if (!next(value)) return false;
            opt.outSummary = value;
        }
        else if (flag == "--lights")
        {
            if (!next(value)) return false;
            opt.lights = static_cast<uint32_t>(std::stoul(value));
        }
        else if (flag == "--hours")
        {
            if (!next(value)) return false;
            opt.hours = static_cast<uint32_t>(std::stoul(value));
        }
        else if (flag == "--seed")
        {
            if (!next(value)) return false;
            opt.seed = std::stoull(value);
        }
        else if (flag == "--producers")
        {
            if (!next(value)) return false;
            opt.producers = std::stoi(value);
        }
        else if (flag == "--consumers")
        {
            if (!next(value)) return false;
            opt.consumers = std::stoi(value);
        }
        else if (flag == "--buffer")
        {
            if (!next(value)) return false;
            opt.bufferBlocks = static_cast<size_t>(std::stoull(value));
        }
        else if (flag == "--block")
        {
            if (!next(value)) return false;
            opt.blockRecords = static_cast<size_t>(std::stoull(value));
        }
        else if (flag == "--summary")
        {
            if (!next(value)) return false;
            opt.summaryName = value;
            if (value == "local")       opt.summary = SummaryMode::Local;
            else if (value == "atomic") opt.summary = SummaryMode::Atomic;
            else if (value == "shared") opt.summary = SummaryMode::Shared;
            else
            {
                error = "unknown summary mode: " + value;
                return false;
            }
        }
        else if (flag == "--top")
        {
            if (!next(value)) return false;
            opt.topN = std::stoi(value);
        }
        else if (flag == "--print-hours")
        {
            if (!next(value)) return false;
            opt.printHours = std::stoi(value);
        }
        else if (flag == "--progress")
        {
            opt.progress = true;
        }
        else if (flag == "--csv")
        {
            opt.csv = true;
        }
        else if (flag == "--help" || flag == "-h")
        {
            usage();
            std::exit(0);
        }
        else
        {
            error = "unknown option: " + flag;
            return false;
        }
    }

    if (opt.mode != "gen" && opt.inPath.empty())
    {
        error = "--in is required";
        return false;
    }
    if (opt.producers < 1 || opt.consumers < 1)
    {
        error = "--producers and --consumers must be at least 1";
        return false;
    }
    if (opt.blockRecords < 1)
    {
        error = "--block must be at least 1";
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Mode: gen
// ---------------------------------------------------------------------------

void appendUInt(std::vector<char>& buf, uint64_t v)
{
    char tmp[24];
    int n = 0;
    if (v == 0)
    {
        tmp[n++] = '0';
    }
    while (v > 0)
    {
        tmp[n++] = static_cast<char>('0' + (v % 10));
        v /= 10;
    }
    while (n > 0)
    {
        buf.push_back(tmp[--n]);
    }
}

int runGenerate(const Options& opt)
{
    if (opt.lights == 0 || opt.hours == 0)
    {
        std::cerr << "gen needs --lights and --hours\n";
        return 1;
    }

    const uint32_t lights = opt.lights;
    const uint32_t hours = opt.hours;
    const long long baseHour = daysFromCivil(2026, 8, 1) * 24;  // 2026-08-01 00:00

    // Each light gets a fixed base rate, and one light in every 89 is a busy
    // intersection. Rush hour weights push the morning and evening peaks up so
    // the top N list changes across the day instead of being flat noise.
    static const int rushWeight[24] =
    {
        30, 22, 18, 16, 20, 35, 60, 90, 100, 85, 70, 68,
        72, 70, 68, 75, 92, 100, 88, 70, 58, 48, 40, 34
    };

    std::vector<uint32_t> baseRate(lights);
    for (uint32_t l = 0; l < lights; ++l)
    {
        uint32_t base = 18 + (l % 7) * 6;
        if (l % 89 == 0)
        {
            base *= 3;
        }
        baseRate[l] = base;
    }

    std::ofstream out(opt.outPath, std::ios::binary);
    if (!out)
    {
        std::cerr << "cannot open output file: " << opt.outPath << "\n";
        return 1;
    }

    std::mt19937_64 rng(opt.seed);
    std::vector<char> buf;
    buf.reserve(8u << 20);

    const auto start = Clock::now();
    uint64_t lines = 0;

    for (uint32_t h = 0; h < hours; ++h)
    {
        const long long absHour = baseHour + h;
        const int hourOfDay = static_cast<int>(absHour % 24);
        const int weight = rushWeight[hourOfDay];

        int y = 0;
        unsigned mo = 0;
        unsigned da = 0;
        civilFromDays(absHour / 24, y, mo, da);

        for (int slot = 0; slot < 12; ++slot)
        {
            char stamp[20];
            std::snprintf(stamp, sizeof(stamp), "%04d-%02u-%02u %02d:%02d",
                          y, mo, da, hourOfDay, slot * 5);
            const size_t stampLen = std::strlen(stamp);

            for (uint32_t l = 0; l < lights; ++l)
            {
                const long long mean = (static_cast<long long>(baseRate[l]) * weight) / 100;
                const long long spread = mean / 3 + 4;
                long long value = mean + static_cast<long long>(rng() % static_cast<uint64_t>(2 * spread + 1)) - spread;
                if (value < 0)
                {
                    value = 0;  // a 5 minute window cannot record a negative count
                }
                const uint32_t cars = static_cast<uint32_t>(value);

                buf.insert(buf.end(), stamp, stamp + stampLen);
                buf.push_back(',');
                buf.push_back('T');
                buf.push_back('L');
                // zero pad to four digits so the file lines up when you look at it
                if (l < 1000) buf.push_back('0');
                if (l < 100)  buf.push_back('0');
                if (l < 10)   buf.push_back('0');
                appendUInt(buf, l);
                buf.push_back(',');
                appendUInt(buf, cars);
                buf.push_back('\n');
                ++lines;
            }

            if (buf.size() > (7u << 20))
            {
                out.write(buf.data(), static_cast<std::streamsize>(buf.size()));
                buf.clear();
            }
        }
    }

    if (!buf.empty())
    {
        out.write(buf.data(), static_cast<std::streamsize>(buf.size()));
    }
    out.close();

    std::ifstream check(opt.outPath, std::ios::binary | std::ios::ate);
    const long long bytes = check ? static_cast<long long>(check.tellg()) : 0;

    std::cout << "Wrote " << opt.outPath << "\n"
              << "  lights            : " << lights << "\n"
              << "  hours             : " << hours << "\n"
              << "  rows              : " << lines << "  (lights x 12 x hours)\n"
              << "  size              : " << (bytes / (1024.0 * 1024.0)) << " MiB\n"
              << "  elapsed           : " << msSince(start) << " ms\n"
              << "\nRun it with:\n"
              << "  ./traffic_sim seq --in " << opt.outPath << "\n"
              << "  ./traffic_sim par --in " << opt.outPath
              << " --producers 4 --consumers 4 --buffer 64 --block 4096\n";
    return 0;
}

// ---------------------------------------------------------------------------
// Shared state for the parallel run
// ---------------------------------------------------------------------------

struct RunContext
{
    BoundedBuffer* buffer = nullptr;
    SummaryStore* store = nullptr;

    long long baseHour = 0;
    uint32_t lights = 0;
    uint32_t hours = 0;

    std::atomic<uint64_t> recordsProduced{0};
    std::atomic<uint64_t> recordsConsumed{0};
    std::atomic<bool> failed{false};

    std::mutex errorMutex;
    std::string errorText;

    void fail(const std::string& message)
    {
        {
            std::lock_guard<std::mutex> lock(errorMutex);
            if (errorText.empty())
            {
                errorText = message;
            }
        }
        failed.store(true, std::memory_order_relaxed);
        buffer->stop();
    }
};

// ---------------------------------------------------------------------------
// Producer
// ---------------------------------------------------------------------------
//
// Each producer owns the byte range [start, end). A line belongs to the range
// that contains its first byte. To find its own first whole line a producer
// looks at the byte just before start: if that byte is a newline it is already
// sitting on a line boundary, otherwise it throws away the partial line it
// landed in the middle of. The producer then keeps reading while the current
// line starts before end, which means it finishes the line that straddles the
// boundary. Every line is read exactly once.

void producerThread(int id, const std::string& path, uint64_t start, uint64_t end,
                    size_t blockRecords, RunContext& ctx)
{
    (void)id;

    // A 1 MiB stream buffer cuts the read syscall count by about 128x compared
    // with the default, which matters once the file runs to hundreds of MiB.
    std::vector<char> streamBuf(1u << 20);
    std::ifstream in;
    in.rdbuf()->pubsetbuf(streamBuf.data(), static_cast<std::streamsize>(streamBuf.size()));
    in.open(path, std::ios::binary);
    if (!in)
    {
        ctx.fail("producer could not open " + path);
        return;
    }

    uint64_t pos = start;
    if (start > 0)
    {
        in.seekg(static_cast<std::streamoff>(start - 1), std::ios::beg);
        char prev = 0;
        in.read(&prev, 1);
        if (!in)
        {
            return;
        }
        if (prev != '\n')
        {
            std::string discard;
            if (!std::getline(in, discard))
            {
                return;
            }
            pos = start + discard.size() + 1;
        }
    }
    else
    {
        in.seekg(0, std::ios::beg);
    }

    Block block;
    block.reserve(blockRecords);

    std::string line;
    uint64_t produced = 0;

    while (pos < end)
    {
        if (!std::getline(in, line))
        {
            break;
        }
        const uint64_t consumedBytes = line.size() + 1;

        if (!line.empty() && line.back() == '\r')
        {
            line.pop_back();
        }
        if (!line.empty())
        {
            long long absHour = 0;
            uint32_t light = 0;
            uint32_t cars = 0;

            if (!parseLine(line.c_str(), line.size(), absHour, light, cars))
            {
                ctx.fail("malformed line at byte " + std::to_string(pos) + ": " + line);
                return;
            }

            const long long hourIdx = absHour - ctx.baseHour;
            if (hourIdx < 0 || hourIdx >= static_cast<long long>(ctx.hours) || light >= ctx.lights)
            {
                ctx.fail("record outside the probed range (hour index " + std::to_string(hourIdx) +
                         ", light " + std::to_string(light) + "). Pass --lights and --hours to size "
                         "the summary manually.");
                return;
            }

            block.push_back(Record{static_cast<uint32_t>(hourIdx), light, cars});
            ++produced;

            if (block.size() >= blockRecords)
            {
                if (!ctx.buffer->push(std::move(block)))
                {
                    return;  // someone called stop()
                }
                block = Block();
                block.reserve(blockRecords);
            }
        }

        pos += consumedBytes;
    }

    if (!block.empty())
    {
        if (!ctx.buffer->push(std::move(block)))
        {
            return;
        }
    }

    ctx.recordsProduced.fetch_add(produced, std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// Consumer
// ---------------------------------------------------------------------------

void consumerThread(int id, RunContext& ctx)
{
    Block block;
    uint64_t consumed = 0;

    while (ctx.buffer->pop(block))
    {
        for (const Record& r : block)
        {
            applyRecord(*ctx.store, id, r);
        }
        consumed += block.size();
        ctx.recordsConsumed.fetch_add(block.size(), std::memory_order_relaxed);
        block.clear();
    }

    (void)consumed;
}

// ---------------------------------------------------------------------------
// Mode: seq
// ---------------------------------------------------------------------------

int runSequential(const Options& opt)
{
    FileProbe probe;
    std::string error;
    if (!probeFile(opt.inPath, probe, error))
    {
        std::cerr << error << "\n";
        return 1;
    }

    SummaryStore store;
    store.mode = SummaryMode::Local;
    store.lights = opt.lights ? opt.lights : probe.lastLight + 1;
    store.hours = opt.hours ? opt.hours
                            : static_cast<uint32_t>(probe.lastHour - probe.baseHour + 1);
    store.global.assign(store.cells(), 0);

    std::cout << "Sequential run\n"
              << "  input             : " << opt.inPath << " ("
              << (probe.size / (1024.0 * 1024.0)) << " MiB)\n"
              << "  lights x hours    : " << store.lights << " x " << store.hours << "\n";

    std::vector<char> streamBuf(1u << 20);
    std::ifstream in;
    in.rdbuf()->pubsetbuf(streamBuf.data(), static_cast<std::streamsize>(streamBuf.size()));
    in.open(opt.inPath, std::ios::binary);
    if (!in)
    {
        std::cerr << "cannot open " << opt.inPath << "\n";
        return 1;
    }

    const auto totalStart = Clock::now();
    const auto ingestStart = Clock::now();

    std::string line;
    uint64_t records = 0;

    while (std::getline(in, line))
    {
        if (!line.empty() && line.back() == '\r')
        {
            line.pop_back();
        }
        if (line.empty())
        {
            continue;
        }

        long long absHour = 0;
        uint32_t light = 0;
        uint32_t cars = 0;
        if (!parseLine(line.c_str(), line.size(), absHour, light, cars))
        {
            std::cerr << "malformed line: " << line << "\n";
            return 1;
        }

        const long long hourIdx = absHour - probe.baseHour;
        if (hourIdx < 0 || hourIdx >= static_cast<long long>(store.hours) || light >= store.lights)
        {
            std::cerr << "record outside the probed range. Pass --lights and --hours.\n";
            return 1;
        }

        store.global[static_cast<size_t>(hourIdx) * store.lights + light] += cars;
        ++records;
    }

    const double ingestMs = msSince(ingestStart);

    double rankMs = 0;
    rankAndReport(store, probe.baseHour, opt.topN, opt.printHours, opt.outSummary, rankMs);

    const double totalMs = msSince(totalStart);
    const uint64_t checksum = summaryChecksum(store);

    std::cout << "\n  records           : " << records << "\n"
              << "  read and count    : " << ingestMs << " ms\n"
              << "  rank              : " << rankMs << " ms\n"
              << "  total             : " << totalMs << " ms\n"
              << "  checksum          : " << checksum << "\n";

    if (opt.csv)
    {
        std::cout << "CSV,seq,1,0,0,0,none," << records << "," << totalMs << "," << checksum << "\n";
    }
    return 0;
}

// ---------------------------------------------------------------------------
// Mode: par
// ---------------------------------------------------------------------------

int runParallel(const Options& opt)
{
    FileProbe probe;
    std::string error;
    if (!probeFile(opt.inPath, probe, error))
    {
        std::cerr << error << "\n";
        return 1;
    }

    const uint32_t lights = opt.lights ? opt.lights : probe.lastLight + 1;
    const uint32_t hours = opt.hours ? opt.hours
                                     : static_cast<uint32_t>(probe.lastHour - probe.baseHour + 1);

    SummaryStore store;
    store.mode = opt.summary;
    store.lights = lights;
    store.hours = hours;

    const size_t cells = store.cells();
    if (store.mode == SummaryMode::Local)
    {
        store.locals.resize(static_cast<size_t>(opt.consumers));
        for (auto& local : store.locals)
        {
            local.assign(cells, 0);
        }
    }
    else if (store.mode == SummaryMode::Atomic)
    {
        store.atomics.reset(new std::atomic<uint64_t>[cells]);
        for (size_t i = 0; i < cells; ++i)
        {
            store.atomics[i].store(0, std::memory_order_relaxed);
        }
    }
    else
    {
        store.global.assign(cells, 0);
    }

    BoundedBuffer buffer(opt.bufferBlocks);

    RunContext ctx;
    ctx.buffer = &buffer;
    ctx.store = &store;
    ctx.baseHour = probe.baseHour;
    ctx.lights = lights;
    ctx.hours = hours;

    std::cout << "Parallel run\n"
              << "  input             : " << opt.inPath << " ("
              << (probe.size / (1024.0 * 1024.0)) << " MiB)\n"
              << "  lights x hours    : " << lights << " x " << hours << "\n"
              << "  producers         : " << opt.producers << "\n"
              << "  consumers         : " << opt.consumers << "\n"
              << "  buffer capacity   : " << buffer.capacity() << " blocks\n"
              << "  block size        : " << opt.blockRecords << " records\n"
              << "  summary mode      : " << opt.summaryName << "\n";

    const auto totalStart = Clock::now();
    const auto ingestStart = Clock::now();

    std::atomic<bool> running{true};
    std::thread monitor;
    if (opt.progress)
    {
        monitor = std::thread([&]()
        {
            int ticks = 0;
            while (running.load(std::memory_order_relaxed))
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(250));
                if (++ticks % 20 != 0)
                {
                    continue;
                }
                uint64_t pushed = 0, popped = 0, pw = 0, cw = 0;
                size_t depth = 0;
                buffer.stats(pushed, popped, pw, cw, depth);
                std::cout << "  [" << std::fixed << std::setprecision(1)
                          << (msSince(ingestStart) / 1000.0) << "s] consumed "
                          << ctx.recordsConsumed.load(std::memory_order_relaxed)
                          << " records, blocks " << popped << "/" << pushed
                          << ", queue depth " << depth << "\n" << std::flush;
            }
        });
    }

    std::vector<std::thread> producers;
    producers.reserve(static_cast<size_t>(opt.producers));
    for (int i = 0; i < opt.producers; ++i)
    {
        const uint64_t start = probe.size * static_cast<uint64_t>(i) / static_cast<uint64_t>(opt.producers);
        const uint64_t end = probe.size * static_cast<uint64_t>(i + 1) / static_cast<uint64_t>(opt.producers);
        producers.emplace_back(producerThread, i, opt.inPath, start, end, opt.blockRecords, std::ref(ctx));
    }

    std::vector<std::thread> consumers;
    consumers.reserve(static_cast<size_t>(opt.consumers));
    for (int i = 0; i < opt.consumers; ++i)
    {
        consumers.emplace_back(consumerThread, i, std::ref(ctx));
    }

    for (auto& t : producers)
    {
        t.join();
    }
    buffer.close();
    for (auto& t : consumers)
    {
        t.join();
    }

    running.store(false, std::memory_order_relaxed);
    if (monitor.joinable())
    {
        monitor.join();
    }

    const double ingestMs = msSince(ingestStart);

    if (ctx.failed.load(std::memory_order_relaxed))
    {
        std::lock_guard<std::mutex> lock(ctx.errorMutex);
        std::cerr << "run aborted: " << ctx.errorText << "\n";
        return 1;
    }

    const auto mergeStart = Clock::now();
    finaliseSummary(store, opt.consumers);
    const double mergeMs = msSince(mergeStart);

    double rankMs = 0;
    rankAndReport(store, probe.baseHour, opt.topN, opt.printHours, opt.outSummary, rankMs);

    const double totalMs = msSince(totalStart);
    const uint64_t checksum = summaryChecksum(store);

    uint64_t pushed = 0, popped = 0, pushWaits = 0, popWaits = 0;
    size_t maxDepth = 0;
    buffer.stats(pushed, popped, pushWaits, popWaits, maxDepth);

    std::cout << "\n  records produced  : " << ctx.recordsProduced.load() << "\n"
              << "  records consumed  : " << ctx.recordsConsumed.load() << "\n"
              << "  blocks pushed     : " << pushed << "\n"
              << "  blocks popped     : " << popped << "\n"
              << "  producer waits    : " << pushWaits << "  (queue was full)\n"
              << "  consumer waits    : " << popWaits << "  (queue was empty)\n"
              << "  peak queue depth  : " << maxDepth << " / " << buffer.capacity() << "\n"
              << "  produce + consume : " << ingestMs << " ms\n"
              << "  merge             : " << mergeMs << " ms\n"
              << "  rank              : " << rankMs << " ms\n"
              << "  total             : " << totalMs << " ms\n"
              << "  checksum          : " << checksum << "\n";

    if (opt.csv)
    {
        std::cout << "CSV,par," << opt.producers << "," << opt.consumers << ","
                  << opt.bufferBlocks << "," << opt.blockRecords << "," << opt.summaryName << ","
                  << ctx.recordsConsumed.load() << "," << totalMs << "," << checksum << "\n";
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv)
{
    std::ios::sync_with_stdio(false);

    Options opt;
    std::string error;
    if (!parseArgs(argc, argv, opt, error))
    {
        std::cerr << error << "\n\n";
        usage();
        return 1;
    }

    if (opt.mode == "gen")
    {
        return runGenerate(opt);
    }
    if (opt.mode == "seq")
    {
        return runSequential(opt);
    }
    return runParallel(opt);
}