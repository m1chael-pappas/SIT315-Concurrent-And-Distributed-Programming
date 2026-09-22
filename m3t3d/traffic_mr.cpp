/*
 * SIT315 Module 3 - Task M3.T3D - Traffic Control Simulator, MPI MapReduce
 * Michael Pappas
 *
 * The same question as M2.T3D, which traffic lights were the most congested in
 * each hour, answered by MPI processes that share one input file instead of
 * threads that share one queue.
 *
 *   map      every rank reads its own byte range of the file with MPI-IO and
 *            combines its records into partial totals per (hour, light)
 *   plan     the ranks agree on which reducer owns which key
 *   shuffle  one MPI_Alltoallv_c moves every partial total to its owner
 *   reduce   every reducer sums the partials for its keys and keeps the top N
 *            lights of each hour it holds
 *   gather   rank 0 collects the per hour candidates and prints the ranking
 *
 * Build:
 *   MPICH_CXX=g++ ~/anaconda3/bin/mpic++ -std=c++17 -O2 -Wall -Wextra -o traffic_mr traffic_mr.cpp
 *
 * Run:
 *   ~/anaconda3/bin/mpirun -np 3 ./traffic_mr --in tiny.csv --top 3 --print-hours 2 --trace
 *   ~/anaconda3/bin/mpirun -np 8 ./traffic_mr --in ../m2t3d/data/traffic_large.csv --partition weighted
 */

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <new>
#include <queue>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "traffic_common.h"

namespace
{

using traffic::LightTotal;

/** Bytes requested from MPI-IO per read while streaming a rank's range. */
constexpr MPI_Offset kChunkBytes = MPI_Offset{16} << 20;

/** Longest line accepted. A partial line longer than this means the file is not in the expected format. */
constexpr size_t kMaxLine = 4096;

/** A partial or final car total for one (hour, light) key, and the only record type that crosses the network. */
struct KeyTotal
{
    uint32_t hour;
    uint32_t light;
    uint64_t cars;
};

/** Packs a key into one integer that orders by hour, then by light. */
inline uint64_t packKey(uint32_t hour, uint32_t light)
{
    return (static_cast<uint64_t>(hour) << 32) | light;
}

/**
 * SplitMix64 finaliser. Hashing the hour before taking it modulo the reducer
 * count avoids the aliasing that plain hour % R has against the 24 hour cycle
 * whenever R divides 24.
 */
inline uint64_t mix64(uint64_t x)
{
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

/**
 * How keys are assigned to reducers.
 *
 *   Block     equal runs of consecutive hours, blind to how much data each hour holds
 *   Hash      a hash of the hour, the MapReduce default partitioner
 *   Weighted  consecutive hours cut at equal shares of a global per hour histogram
 *   Light     a hash of (hour, light), so one heavy hour spreads over every reducer
 *   Hybrid    Light for any hour too heavy for one reducer, Weighted for the rest
 */
enum class Partition
{
    Block,
    Hash,
    Weighted,
    Light,
    Hybrid
};

inline const char* partitionName(Partition p)
{
    switch (p)
    {
        case Partition::Block:    return "block";
        case Partition::Hash:     return "hash";
        case Partition::Weighted: return "weighted";
        case Partition::Light:    return "light";
        case Partition::Hybrid:   return "hybrid";
    }
    return "?";
}

struct Options
{
    std::string inPath;
    std::string outSummary;
    Partition partition = Partition::Weighted;
    int reducers = 0;
    int topN = 5;
    int printHours = 3;
    bool combine = true;
    bool verify = false;
    bool trace = false;
    bool matrix = false;
    bool csv = false;
    bool help = false;
};

void usage()
{
    std::cout <<
        "SIT315 M3.T3D traffic control simulator, MPI MapReduce\n"
        "\n"
        "  mpirun -np P ./traffic_mr --in FILE [--partition block|hash|weighted|light|hybrid]\n"
        "                            [--reducers R] [--top N] [--print-hours K]\n"
        "                            [--out-summary FILE] [--no-combine] [--verify]\n"
        "                            [--trace] [--matrix] [--csv]\n"
        "\n"
        "  --partition   how keys are assigned to reducers (default weighted)\n"
        "  --reducers    how many of the P ranks reduce, 1..P (default P)\n"
        "  --no-combine  ship every record instead of per-key partial totals\n"
        "  --verify      gather every total to rank 0 and print the m2t3d checksum\n"
        "  --trace       print what every rank mapped, sent and reduced, for small files\n"
        "  --matrix      print the P x P shuffle matrix\n";
}

bool parseArgs(int argc, char** argv, Options& opt, std::string& error)
{
    for (int i = 1; i < argc; ++i)
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
        else if (flag == "--out-summary")
        {
            if (!next(value)) return false;
            opt.outSummary = value;
        }
        else if (flag == "--partition")
        {
            if (!next(value)) return false;
            if (value == "block")         opt.partition = Partition::Block;
            else if (value == "hash")     opt.partition = Partition::Hash;
            else if (value == "weighted") opt.partition = Partition::Weighted;
            else if (value == "light")    opt.partition = Partition::Light;
            else if (value == "hybrid")   opt.partition = Partition::Hybrid;
            else
            {
                error = "unknown partition: " + value;
                return false;
            }
        }
        else if (flag == "--reducers")
        {
            if (!next(value)) return false;
            opt.reducers = std::stoi(value);
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
        else if (flag == "--no-combine")
        {
            opt.combine = false;
        }
        else if (flag == "--verify")
        {
            opt.verify = true;
        }
        else if (flag == "--trace")
        {
            opt.trace = true;
        }
        else if (flag == "--matrix")
        {
            opt.matrix = true;
        }
        else if (flag == "--csv")
        {
            opt.csv = true;
        }
        else if (flag == "--help" || flag == "-h")
        {
            opt.help = true;
            return true;
        }
        else
        {
            error = "unknown option: " + flag;
            return false;
        }
    }

    if (opt.inPath.empty())
    {
        error = "--in is required";
        return false;
    }
    if (opt.topN < 1)
    {
        error = "--top must be at least 1";
        return false;
    }
    return true;
}

/**
 * Builds the MPI datatype for KeyTotal from its real field offsets, then resizes
 * it to sizeof(KeyTotal) so arrays of it step correctly past any padding.
 */
MPI_Datatype makeKeyTotalType()
{
    const int lengths[3] = {1, 1, 1};
    const MPI_Aint offsets[3] =
    {
        static_cast<MPI_Aint>(offsetof(KeyTotal, hour)),
        static_cast<MPI_Aint>(offsetof(KeyTotal, light)),
        static_cast<MPI_Aint>(offsetof(KeyTotal, cars))
    };
    const MPI_Datatype types[3] = {MPI_UINT32_T, MPI_UINT32_T, MPI_UINT64_T};

    MPI_Datatype raw = MPI_DATATYPE_NULL;
    MPI_Datatype resized = MPI_DATATYPE_NULL;
    MPI_Type_create_struct(3, lengths, offsets, types, &raw);
    MPI_Type_create_resized(raw, 0, static_cast<MPI_Aint>(sizeof(KeyTotal)), &resized);
    MPI_Type_commit(&resized);
    MPI_Type_free(&raw);
    return resized;
}

/** Prints a message from any rank and takes every rank down with it. */
[[noreturn]] void fatal(int rank, const std::string& message)
{
    std::fprintf(stderr, "rank %d: %s\n", rank, message.c_str());
    std::fflush(stderr);
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::exit(1);
}

/** Collects one string from every rank onto rank 0, in rank order. Collective. */
std::vector<std::string> gatherText(const std::string& mine, int rank, int procs)
{
    const int len = static_cast<int>(mine.size());
    std::vector<int> lens(rank == 0 ? procs : 0);
    MPI_Gather(&len, 1, MPI_INT, lens.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<int> displs(lens.size(), 0);
    int total = 0;
    for (size_t r = 0; r < lens.size(); ++r)
    {
        displs[r] = total;
        total += lens[r];
    }

    std::vector<char> all(static_cast<size_t>(total));
    MPI_Gatherv(mine.data(), len, MPI_CHAR, all.data(), lens.data(), displs.data(), MPI_CHAR,
                0, MPI_COMM_WORLD);

    std::vector<std::string> out;
    for (size_t r = 0; r < lens.size(); ++r)
    {
        out.emplace_back(all.data() + displs[r], static_cast<size_t>(lens[r]));
    }
    return out;
}

inline std::string lightLabel(uint32_t light)
{
    char buf[16];
    std::snprintf(buf, sizeof(buf), "TL%04u", light);
    return std::string(buf);
}

// ---------------------------------------------------------------------------
// Map
// ---------------------------------------------------------------------------

/** Everything one rank knows after mapping its byte range. */
struct MapResult
{
    MPI_Offset start = 0;
    MPI_Offset end = 0;
    uint64_t records = 0;
    uint64_t cars = 0;
    uint32_t minHour = std::numeric_limits<uint32_t>::max();
    uint32_t maxHour = 0;
    double workMs = 0;
    std::vector<KeyTotal> entries;
};

/**
 * Streams the lines whose first byte lies in [start, end), in order.
 *
 * Applies the m2t3d rule for splitting a file by bytes: a line belongs to the
 * range holding its first byte. If the byte before start is not a newline the
 * rank landed mid-line, so it drops that fragment because the previous rank
 * owns it. Reading continues past end until the line that straddles end is
 * complete, so every line is read exactly once across all ranks.
 *
 * Reads stop at end, then continue in kMaxLine steps only as far as that last
 * line needs, so a rank never reads another rank's share.
 */
template <typename OnLine>
void forEachLine(MPI_File fh, MPI_Offset fileSize, MPI_Offset start, MPI_Offset end,
                 int rank, OnLine onLine)
{
    if (start >= end)
    {
        return;
    }

    bool skipping = false;
    if (start > 0)
    {
        char prev = 0;
        MPI_Status status;
        if (MPI_File_read_at(fh, start - 1, &prev, 1, MPI_CHAR, &status) != MPI_SUCCESS)
        {
            fatal(rank, "MPI_File_read_at failed at byte " + std::to_string(start - 1));
        }
        skipping = prev != '\n';
    }

    std::vector<char> buf;
    MPI_Offset bufOffset = start;
    MPI_Offset readPos = start;

    while (true)
    {
        const size_t carried = buf.size();
        if (readPos < fileSize)
        {
            const MPI_Offset wanted = std::max(end - readPos, static_cast<MPI_Offset>(kMaxLine));
            const MPI_Offset want = std::min({kChunkBytes, fileSize - readPos, wanted});
            buf.resize(carried + static_cast<size_t>(want));
            MPI_Status status;
            if (MPI_File_read_at_c(fh, readPos, buf.data() + carried, static_cast<MPI_Count>(want),
                                   MPI_CHAR, &status) != MPI_SUCCESS)
            {
                fatal(rank, "MPI_File_read_at_c failed at byte " + std::to_string(readPos));
            }
            MPI_Count got = 0;
            MPI_Get_count_c(&status, MPI_CHAR, &got);
            if (got != want)
            {
                fatal(rank, "short read at byte " + std::to_string(readPos));
            }
            readPos += want;
        }
        const bool eof = readPos >= fileSize;

        size_t lineBegin = 0;
        while (lineBegin < buf.size())
        {
            const char* p = buf.data() + lineBegin;
            const void* nl = std::memchr(p, '\n', buf.size() - lineBegin);
            if (nl == nullptr)
            {
                break;
            }
            const size_t len = static_cast<size_t>(static_cast<const char*>(nl) - p);
            const MPI_Offset lineStart = bufOffset + static_cast<MPI_Offset>(lineBegin);

            if (skipping)
            {
                skipping = false;
            }
            else if (lineStart >= end)
            {
                return;
            }
            else
            {
                onLine(p, (len > 0 && p[len - 1] == '\r') ? len - 1 : len, lineStart);
            }
            lineBegin += len + 1;
        }

        const size_t rest = buf.size() - lineBegin;
        if (eof)
        {
            const MPI_Offset lineStart = bufOffset + static_cast<MPI_Offset>(lineBegin);
            if (rest > 0 && !skipping && lineStart < end)
            {
                const char* p = buf.data() + lineBegin;
                onLine(p, (p[rest - 1] == '\r') ? rest - 1 : rest, lineStart);
            }
            return;
        }
        if (rest > kMaxLine)
        {
            fatal(rank, "line longer than " + std::to_string(kMaxLine) + " bytes near byte " +
                        std::to_string(bufOffset + static_cast<MPI_Offset>(lineBegin)));
        }

        std::memmove(buf.data(), buf.data() + lineBegin, rest);
        buf.resize(rest);
        bufOffset += static_cast<MPI_Offset>(lineBegin);
    }
}

/** Largest light id the dense per hour rows accept, which caps one row at 128 MiB. */
constexpr uint32_t kMaxLight = 1u << 24;

/**
 * Car totals per (hour, light): one dense row of totals per hour, indexed by
 * light id, created the first time an hour appears and grown as higher ids
 * arrive. Hours may come in any order and any range. Light ids are small dense
 * integers, the same assumption m2t3d's hours x lights array made.
 *
 * This is the combiner. It suits the map side because a mapper reads whole
 * runs of consecutive hours, so its rows are dense. A reducer under the light
 * partition holds only 1/R of each hour's lights, so it merges sorted runs
 * instead, see reduceReceived.
 *
 * Two other containers were measured and replaced. std::unordered_map
 * allocates a node per key, and on shuffled input, about 900 thousand keys per
 * rank, freeing them stalled the next larger allocation for over 100 ms while
 * glibc consolidated its free lists. A flat open-addressing table fixed that
 * but hashed each hour's lights to random slots, which cost 35% on
 * time-ordered input where every light of the current hour is updated twelve
 * times in a row. Rows keep those updates sequential.
 */
class HourTable
{
public:
    void add(uint32_t hour, uint32_t light, uint64_t cars)
    {
        std::vector<uint64_t>& row = rowFor(hour);
        if (light >= row.size())
        {
            row.resize(std::max<size_t>(static_cast<size_t>(light) + 1, row.size() * 2));
        }
        row[light] += cars;
    }

    /** Calls visit(hour, light, total) for every non-zero total, in hour then light order. */
    template <typename Visit>
    void forEach(Visit visit) const
    {
        std::vector<uint32_t> hours;
        hours.reserve(rows_.size());
        for (const auto& kv : rows_)
        {
            hours.push_back(kv.first);
        }
        std::sort(hours.begin(), hours.end());

        for (uint32_t hour : hours)
        {
            const std::vector<uint64_t>& row = rows_.at(hour);
            for (size_t light = 0; light < row.size(); ++light)
            {
                if (row[light] != 0)
                {
                    visit(hour, static_cast<uint32_t>(light), row[light]);
                }
            }
        }
    }

    size_t nonZero() const
    {
        size_t n = 0;
        for (const auto& kv : rows_)
        {
            n += static_cast<size_t>(std::count_if(kv.second.begin(), kv.second.end(),
                                                   [](uint64_t v) { return v != 0; }));
        }
        return n;
    }

private:
    std::vector<uint64_t>& rowFor(uint32_t hour)
    {
        if (lastRow_ != nullptr && hour == lastHour_)
        {
            return *lastRow_;
        }
        lastHour_ = hour;
        lastRow_ = &rows_[hour];
        return *lastRow_;
    }

    std::unordered_map<uint32_t, std::vector<uint64_t>> rows_;
    uint32_t lastHour_ = 0;
    std::vector<uint64_t>* lastRow_ = nullptr;
};

/**
 * Map step for one rank: parses its byte range into (hour, light, cars) records.
 * With the combiner on, records are summed per key before anything leaves the
 * rank. With it off, every record is shipped as it is.
 */
void mapRange(MPI_File fh, MPI_Offset fileSize, int rank, int procs, bool combine,
              std::ostringstream* trace, MapResult& out)
{
    out.start = fileSize * rank / procs;
    out.end = fileSize * (rank + 1) / procs;

    HourTable combined;

    forEachLine(fh, fileSize, out.start, out.end, rank,
                [&](const char* s, size_t n, MPI_Offset at)
    {
        if (n == 0)
        {
            return;
        }
        long long absHour = 0;
        uint32_t light = 0;
        uint32_t cars = 0;
        if (!traffic::parseLine(s, n, absHour, light, cars))
        {
            fatal(rank, "malformed line at byte " + std::to_string(at) + ": " + std::string(s, n));
        }
        if (absHour < 0 || absHour >= std::numeric_limits<uint32_t>::max())
        {
            fatal(rank, "timestamp out of range at byte " + std::to_string(at));
        }
        const uint32_t hour = static_cast<uint32_t>(absHour);
        if (light >= kMaxLight)
        {
            fatal(rank, "light id " + std::to_string(light) + " at byte " + std::to_string(at) +
                        " is above the dense row limit of " + std::to_string(kMaxLight));
        }

        out.minHour = std::min(out.minHour, hour);
        out.maxHour = std::max(out.maxHour, hour);
        ++out.records;
        out.cars += cars;

        if (trace != nullptr)
        {
            *trace << "    byte " << std::setw(5) << at << "  " << std::string(s, n) << "\n";
        }

        if (combine)
        {
            combined.add(hour, light, cars);
        }
        else
        {
            out.entries.push_back(KeyTotal{hour, light, cars});
        }
    });

    if (combine)
    {
        out.entries.reserve(combined.nonZero());
        combined.forEach([&out](uint32_t hour, uint32_t light, uint64_t cars)
        {
            out.entries.push_back(KeyTotal{hour, light, cars});
        });
    }
}

// ---------------------------------------------------------------------------
// Plan
// ---------------------------------------------------------------------------

/** Marks an hour in ownerOfHour whose keys are spread across every reducer by light. */
constexpr int kSplitHour = -1;

/** Decides which reducer owns a key. Every rank builds an identical copy. */
struct Partitioner
{
    Partition kind = Partition::Weighted;
    int reducers = 1;
    uint32_t minHour = 0;
    std::vector<int> ownerOfHour;
    size_t splitHours = 0;

    int destination(const KeyTotal& e) const
    {
        switch (kind)
        {
            case Partition::Block:
            case Partition::Weighted:
                return ownerOfHour[e.hour - minHour];
            case Partition::Hash:
                return static_cast<int>(mix64(e.hour) % static_cast<uint64_t>(reducers));
            case Partition::Light:
                return byLight(e);
            case Partition::Hybrid:
            {
                const int owner = ownerOfHour[e.hour - minHour];
                return owner == kSplitHour ? byLight(e) : owner;
            }
        }
        return 0;
    }

    int byLight(const KeyTotal& e) const
    {
        return static_cast<int>(mix64(packKey(e.hour, e.light)) % static_cast<uint64_t>(reducers));
    }
};

/** Equal numbers of consecutive hours per reducer, ignoring how much data each hour holds. */
std::vector<int> blockOwners(size_t span, int reducers)
{
    std::vector<int> owner(span);
    for (size_t h = 0; h < span; ++h)
    {
        owner[h] = static_cast<int>(h * static_cast<size_t>(reducers) / span);
    }
    return owner;
}

/**
 * Consecutive hours per reducer, cut where the cumulative weight crosses each
 * reducer's equal share. An hour goes to the reducer whose share holds the
 * midpoint of that hour's weight, so owners never decrease with the hour and
 * each reducer gets one contiguous run. A single hour heavier than one share
 * still lands on one reducer, which is the case the light partition exists for.
 */
std::vector<int> weightedOwners(const std::vector<uint64_t>& weight, int reducers)
{
    uint64_t total = 0;
    for (uint64_t w : weight)
    {
        total += w;
    }

    std::vector<int> owner(weight.size(), 0);
    if (total == 0)
    {
        return owner;
    }

    const unsigned __int128 r = static_cast<unsigned>(reducers);
    uint64_t before = 0;
    for (size_t h = 0; h < weight.size(); ++h)
    {
        const unsigned __int128 mid = static_cast<unsigned __int128>(before) * 2 + weight[h];
        const uint64_t slot = static_cast<uint64_t>(mid * r / (static_cast<unsigned __int128>(total) * 2));
        owner[h] = static_cast<int>(std::min<uint64_t>(slot, static_cast<uint64_t>(reducers - 1)));
        before += weight[h];
    }
    return owner;
}

/**
 * Weighted cuts for every hour that fits on one reducer, and kSplitHour for
 * every hour that does not.
 *
 * Split hours spread evenly over all R reducers, so each reducer's room for
 * the rest is the remaining weight over R. An hour is split when it is heavier
 * than that room. Hours are checked heaviest first, and each one split shrinks
 * the room for the rest, so an hour that only becomes too heavy once another
 * has been spread is caught too. Everything left is cut exactly as Weighted
 * cuts it, so it stays on the rank that mapped it.
 */
std::vector<int> hybridOwners(const std::vector<uint64_t>& weight, int reducers, size_t& splitHours)
{
    uint64_t room = 0;
    for (uint64_t w : weight)
    {
        room += w;
    }

    std::vector<size_t> heaviest(weight.size());
    for (size_t h = 0; h < heaviest.size(); ++h)
    {
        heaviest[h] = h;
    }
    std::sort(heaviest.begin(), heaviest.end(), [&weight](size_t a, size_t b)
    {
        return weight[a] > weight[b];
    });

    std::vector<uint64_t> kept = weight;
    std::vector<bool> split(weight.size(), false);
    splitHours = 0;
    for (size_t h : heaviest)
    {
        if (static_cast<unsigned __int128>(weight[h]) * static_cast<unsigned>(reducers) <= room)
        {
            break;
        }
        split[h] = true;
        kept[h] = 0;
        room -= weight[h];
        ++splitHours;
    }

    std::vector<int> owner = weightedOwners(kept, reducers);
    for (size_t h = 0; h < owner.size(); ++h)
    {
        if (split[h])
        {
            owner[h] = kSplitHour;
        }
    }
    return owner;
}

/**
 * Sums, across every rank, how many entries each hour is about to ship. That
 * count is what a reducer will actually have to process, so it is the right
 * weight for placing hours. Collective.
 */
std::vector<uint64_t> globalHourWeights(const std::vector<KeyTotal>& entries, uint32_t minHour, size_t span)
{
    std::vector<uint64_t> local(span, 0);
    for (const KeyTotal& e : entries)
    {
        ++local[e.hour - minHour];
    }
    std::vector<uint64_t> global(span, 0);
    MPI_Allreduce(local.data(), global.data(), static_cast<int>(span), MPI_UINT64_T, MPI_SUM,
                  MPI_COMM_WORLD);
    return global;
}

/** Builds the partitioner. Collective for the weighted and hybrid strategies, which need the global histogram. */
Partitioner buildPartitioner(Partition kind, int reducers, uint32_t minHour, uint32_t maxHour,
                             const std::vector<KeyTotal>& entries)
{
    Partitioner p;
    p.kind = kind;
    p.reducers = reducers;
    p.minHour = minHour;

    const size_t span = static_cast<size_t>(maxHour - minHour) + 1;
    if (kind == Partition::Block)
    {
        p.ownerOfHour = blockOwners(span, reducers);
    }
    else if (kind == Partition::Weighted)
    {
        p.ownerOfHour = weightedOwners(globalHourWeights(entries, minHour, span), reducers);
    }
    else if (kind == Partition::Hybrid)
    {
        p.ownerOfHour = hybridOwners(globalHourWeights(entries, minHour, span), reducers, p.splitHours);
    }
    return p;
}

// ---------------------------------------------------------------------------
// Shuffle
// ---------------------------------------------------------------------------

struct ShuffleResult
{
    std::vector<KeyTotal> received;
    std::vector<MPI_Count> sendCounts;
    std::vector<MPI_Count> recvCounts;
    std::vector<MPI_Aint> rdispls;
};

/**
 * Moves every entry to the reducer that owns its key. Counts go first with
 * MPI_Alltoall so every rank can size its receive buffer, then the entries
 * themselves with the large-count MPI_Alltoallv_c. Both calls are blocking
 * collectives: the reduce step cannot start until every partial has arrived.
 */
void shuffle(const std::vector<KeyTotal>& entries, const Partitioner& part, int procs,
             MPI_Datatype type, ShuffleResult& out)
{
    std::vector<int> dest(entries.size());
    out.sendCounts.assign(static_cast<size_t>(procs), 0);
    for (size_t i = 0; i < entries.size(); ++i)
    {
        dest[i] = part.destination(entries[i]);
        ++out.sendCounts[static_cast<size_t>(dest[i])];
    }

    std::vector<MPI_Aint> sdispls(static_cast<size_t>(procs), 0);
    for (int r = 1; r < procs; ++r)
    {
        sdispls[r] = sdispls[r - 1] + static_cast<MPI_Aint>(out.sendCounts[r - 1]);
    }

    std::vector<KeyTotal> sendBuf(entries.size());
    std::vector<MPI_Aint> fill = sdispls;
    for (size_t i = 0; i < entries.size(); ++i)
    {
        sendBuf[static_cast<size_t>(fill[dest[i]]++)] = entries[i];
    }

    out.recvCounts.assign(static_cast<size_t>(procs), 0);
    MPI_Alltoall(out.sendCounts.data(), 1, MPI_COUNT, out.recvCounts.data(), 1, MPI_COUNT,
                 MPI_COMM_WORLD);

    out.rdispls.assign(static_cast<size_t>(procs), 0);
    MPI_Count total = 0;
    for (int r = 0; r < procs; ++r)
    {
        out.rdispls[r] = static_cast<MPI_Aint>(total);
        total += out.recvCounts[r];
    }

    out.received.resize(static_cast<size_t>(total));
    MPI_Alltoallv_c(sendBuf.data(), out.sendCounts.data(), sdispls.data(), type,
                    out.received.data(), out.recvCounts.data(), out.rdispls.data(), type,
                    MPI_COMM_WORLD);
}

// ---------------------------------------------------------------------------
// Reduce
// ---------------------------------------------------------------------------

/** A reducer's output. */
struct ReduceResult
{
    std::vector<KeyTotal> candidates;
    uint64_t keys = 0;
};

inline bool keyLess(const KeyTotal& a, const KeyTotal& b)
{
    return packKey(a.hour, a.light) < packKey(b.hour, b.light);
}

/**
 * Calls visit(hour, light, total) for every key with a non-zero total in a
 * buffer already in key order, summing each key's partials as it goes. The
 * reduce step and --verify both read the merged buffer through this.
 */
template <typename Visit>
void forEachTotal(const std::vector<KeyTotal>& sorted, Visit visit)
{
    size_t i = 0;
    while (i < sorted.size())
    {
        const uint32_t hour = sorted[i].hour;
        const uint32_t light = sorted[i].light;
        uint64_t sum = 0;
        while (i < sorted.size() && sorted[i].hour == hour && sorted[i].light == light)
        {
            sum += sorted[i].cars;
            ++i;
        }
        if (sum != 0)
        {
            visit(hour, light, sum);
        }
    }
}

/**
 * Puts the receive buffer in key order by merging the runs the senders
 * delivered. Each sender's segment is already one run in key order: the
 * combiner emits its table in hour then light order, and the counting sort
 * that packs the send buffer is stable. Merging runs pairwise costs n log P
 * instead of a full sort's n log n. With the combiner off a segment holds raw
 * records in file order, so a segment that is not in key order is sorted first.
 */
void mergeRuns(std::vector<KeyTotal>& received, const std::vector<MPI_Aint>& displs,
               const std::vector<MPI_Count>& counts)
{
    std::vector<size_t> bounds;
    for (size_t r = 0; r < displs.size(); ++r)
    {
        if (counts[r] == 0)
        {
            continue;
        }
        const auto first = received.begin() + displs[r];
        const auto last = first + static_cast<std::ptrdiff_t>(counts[r]);
        if (!std::is_sorted(first, last, keyLess))
        {
            std::sort(first, last, keyLess);
        }
        bounds.push_back(static_cast<size_t>(displs[r]));
    }
    bounds.push_back(received.size());

    while (bounds.size() > 2)
    {
        std::vector<size_t> merged;
        for (size_t i = 0; i + 1 < bounds.size(); i += 2)
        {
            merged.push_back(bounds[i]);
            if (i + 2 < bounds.size())
            {
                std::inplace_merge(received.begin() + static_cast<std::ptrdiff_t>(bounds[i]),
                                   received.begin() + static_cast<std::ptrdiff_t>(bounds[i + 1]),
                                   received.begin() + static_cast<std::ptrdiff_t>(bounds[i + 2]),
                                   keyLess);
            }
        }
        merged.push_back(received.size());
        bounds.swap(merged);
    }
}

/**
 * Reduce step for one reducer. Merges the sender runs into key order, then one
 * pass sums each key's partials and feeds a size N min-heap per hour. The
 * heap's top is always the least busy light still in the running, so a new
 * total costs log N and the heap never holds more than N.
 *
 * Totals of zero are skipped, which matches m2t3d leaving empty cells out of
 * both the ranking and the checksum.
 */
void reduceReceived(ShuffleResult& shuffled, int topN, ReduceResult& out)
{
    mergeRuns(shuffled.received, shuffled.rdispls, shuffled.recvCounts);

    using MinHeap = std::priority_queue<LightTotal, std::vector<LightTotal>,
                                        bool (*)(const LightTotal&, const LightTotal&)>;
    MinHeap heap(traffic::busierFirst);
    uint32_t current = 0;

    auto flush = [&]()
    {
        while (!heap.empty())
        {
            out.candidates.push_back(KeyTotal{current, heap.top().light, heap.top().cars});
            heap.pop();
        }
    };

    forEachTotal(shuffled.received, [&](uint32_t hour, uint32_t light, uint64_t sum)
    {
        if (hour != current)
        {
            flush();
            current = hour;
        }
        ++out.keys;
        heap.push(LightTotal{light, sum});
        if (heap.size() > static_cast<size_t>(topN))
        {
            heap.pop();
        }
    });
    flush();
}

// ---------------------------------------------------------------------------
// Gather and report
// ---------------------------------------------------------------------------

/** Gathers a vector of KeyTotal onto rank 0 in rank order. Collective. */
std::vector<KeyTotal> gatherEntries(const std::vector<KeyTotal>& mine, int rank, int procs,
                                    MPI_Datatype type)
{
    const MPI_Count count = static_cast<MPI_Count>(mine.size());
    std::vector<MPI_Count> counts(rank == 0 ? procs : 0);
    MPI_Gather(&count, 1, MPI_COUNT, counts.data(), 1, MPI_COUNT, 0, MPI_COMM_WORLD);

    std::vector<MPI_Aint> displs(counts.size(), 0);
    MPI_Count total = 0;
    for (size_t r = 0; r < counts.size(); ++r)
    {
        displs[r] = static_cast<MPI_Aint>(total);
        total += counts[r];
    }

    std::vector<KeyTotal> all(static_cast<size_t>(total));
    MPI_Gatherv_c(mine.data(), count, type, all.data(), counts.data(), displs.data(), type,
                  0, MPI_COMM_WORLD);
    return all;
}

/**
 * Final ranking on rank 0. With hour-based partitions each hour has at most N
 * candidates from its one owner. With the light partition an hour has up to N
 * from every reducer, and because each light's total is complete on exactly one
 * reducer, the true top N is always inside that union.
 */
std::vector<KeyTotal> mergeCandidates(std::vector<KeyTotal> all, int topN)
{
    std::sort(all.begin(), all.end(), [](const KeyTotal& a, const KeyTotal& b)
    {
        if (a.hour != b.hour)
        {
            return a.hour < b.hour;
        }
        return traffic::busierFirst(LightTotal{a.light, a.cars}, LightTotal{b.light, b.cars});
    });

    std::vector<KeyTotal> ranked;
    size_t i = 0;
    while (i < all.size())
    {
        const uint32_t hour = all[i].hour;
        int kept = 0;
        while (i < all.size() && all[i].hour == hour)
        {
            if (kept < topN)
            {
                ranked.push_back(all[i]);
                ++kept;
            }
            ++i;
        }
    }
    return ranked;
}

/**
 * Prints the first printHours hours in the m2t3d console format and writes
 * every hour to outPath in the m2t3d --out-summary format, so the two files
 * compare byte for byte.
 */
void report(const std::vector<KeyTotal>& ranked, int topN, int printHours, const std::string& outPath)
{
    std::ofstream out;
    if (!outPath.empty())
    {
        out.open(outPath, std::ios::binary);
    }

    std::ostringstream console;
    int printed = 0;

    size_t i = 0;
    while (i < ranked.size())
    {
        const uint32_t hour = ranked[i].hour;
        const std::string label = traffic::formatHour(hour);
        const bool show = printed < printHours;

        if (show)
        {
            console << "  " << label << "\n";
        }
        if (out.is_open())
        {
            out << label;
        }

        size_t place = 0;
        while (i < ranked.size() && ranked[i].hour == hour)
        {
            if (show)
            {
                console << "    " << std::setw(2) << (place + 1) << ". TL"
                        << std::setw(4) << std::setfill('0') << ranked[i].light
                        << std::setfill(' ') << "   " << std::setw(8) << ranked[i].cars
                        << " cars\n";
            }
            if (out.is_open())
            {
                out << ',' << ranked[i].light << ':' << ranked[i].cars;
            }
            ++place;
            ++i;
        }

        if (out.is_open())
        {
            out << '\n';
        }
        if (show)
        {
            ++printed;
        }
    }

    std::cout << "Top " << topN << " congested lights per hour (first "
              << printHours << (printHours == 1 ? " hour" : " hours") << " shown):\n" << console.str();
}

/**
 * FNV-1a over every non-zero total in hour then light order, with hours
 * counted from the first hour in the file. The same walk as m2t3d's
 * summaryChecksum, so both programs print the same number for the same input.
 */
uint64_t summaryChecksum(std::vector<KeyTotal> totals, uint32_t baseHour)
{
    std::sort(totals.begin(), totals.end(), [](const KeyTotal& a, const KeyTotal& b)
    {
        return packKey(a.hour, a.light) < packKey(b.hour, b.light);
    });

    uint64_t h = 1469598103934665603ull;
    auto mix = [&h](uint64_t value)
    {
        for (int b = 0; b < 8; ++b)
        {
            h ^= (value >> (b * 8)) & 0xFFull;
            h *= 1099511628211ull;
        }
    };

    for (const KeyTotal& t : totals)
    {
        mix(t.hour - baseHour);
        mix(t.light);
        mix(t.cars);
    }
    return h;
}


// ---------------------------------------------------------------------------
// Run orchestration
// ---------------------------------------------------------------------------

/** Per rank counters and timings, gathered to rank 0 for the imbalance table. */
struct RankStats
{
    enum Counter { Records, Mapped, SentLocal, SentRemote, Received, Keys, Candidates, NumCounters };
    enum Timer { MapMs, ShuffleMs, ReduceMs, NumTimers };

    uint64_t counter[NumCounters] = {};
    double timer[NumTimers] = {};
};

/** Wall time of each phase on rank 0, measured between barriers so it always includes the slowest rank. */
struct Phases
{
    double map = 0;
    double plan = 0;
    double shuffle = 0;
    double reduce = 0;
    double gather = 0;

    double total() const
    {
        return map + plan + shuffle + reduce + gather;
    }
};

/** Facts every rank agrees on once the map step is done. */
struct GlobalView
{
    uint32_t minHour = 0;
    uint32_t maxHour = 0;
    uint64_t records = 0;
    uint64_t cars = 0;
};

/** Everything rank 0 collects from the other ranks for the summary. */
struct Collected
{
    std::vector<RankStats> stats;
    std::vector<MPI_Offset> starts;
    std::vector<MPI_Offset> ends;
    std::vector<MPI_Count> matrix;
};

/** State shared by every phase of one run on one rank. */
struct Run
{
    int rank = 0;
    int procs = 1;
    int threadLevel = MPI_THREAD_SINGLE;
    const char* phase = "startup";
    Options opt;
    MPI_Datatype keyTotalType = MPI_DATATYPE_NULL;
};

inline const char* threadLevelName(int level)
{
    switch (level)
    {
        case MPI_THREAD_SINGLE:     return "MPI_THREAD_SINGLE";
        case MPI_THREAD_FUNNELED:   return "MPI_THREAD_FUNNELED";
        case MPI_THREAD_SERIALIZED: return "MPI_THREAD_SERIALIZED";
        case MPI_THREAD_MULTIPLE:   return "MPI_THREAD_MULTIPLE";
    }
    return "unknown";
}

/** Max over mean. 1.0 is perfect balance. */
double imbalance(const std::vector<double>& values)
{
    if (values.empty())
    {
        return 1.0;
    }
    double sum = 0;
    double peak = 0;
    for (double v : values)
    {
        sum += v;
        peak = std::max(peak, v);
    }
    const double mean = sum / static_cast<double>(values.size());
    return mean > 0 ? peak / mean : 1.0;
}

/** Rank 0 progress line on stderr. Flushes stdout first so the two streams reach the terminal in order. */
void progress(const Run& run, const std::string& message)
{
    if (run.rank != 0)
    {
        return;
    }
    std::cout.flush();
    std::fflush(stdout);
    std::fprintf(stderr, "  %s\n", message.c_str());
    std::fflush(stderr);
}

/** Ends a phase: waits for every rank, returns the milliseconds since `since`, and restarts it. */
double closePhase(double& since)
{
    MPI_Barrier(MPI_COMM_WORLD);
    const double now = MPI_Wtime();
    const double ms = (now - since) * 1000.0;
    since = now;
    return ms;
}

/** Restarts the phase clock after trace output, so printing never counts towards a phase. */
void resetClock(double& since)
{
    MPI_Barrier(MPI_COMM_WORLD);
    since = MPI_Wtime();
}

/** Parses and checks the command line on every rank. Returns false, with the exit code set, when the run should stop. */
bool configure(int argc, char** argv, Run& run, int& exitCode)
{
    std::string error;
    const bool parsed = parseArgs(argc, argv, run.opt, error);
    if (parsed && !run.opt.help)
    {
        if (run.opt.reducers == 0)
        {
            run.opt.reducers = run.procs;
        }
        if (run.opt.reducers >= 1 && run.opt.reducers <= run.procs)
        {
            return true;
        }
        error = "--reducers must be between 1 and the process count (" + std::to_string(run.procs) + ")";
    }
    if (run.rank == 0)
    {
        if (!error.empty())
        {
            std::cerr << error << "\n\n";
        }
        usage();
    }
    exitCode = error.empty() ? 0 : 1;
    return false;
}

void printBanner(const Run& run)
{
    if (run.rank != 0)
    {
        return;
    }
    char host[MPI_MAX_PROCESSOR_NAME] = {0};
    int hostLen = 0;
    MPI_Get_processor_name(host, &hostLen);

    char lib[MPI_MAX_LIBRARY_VERSION_STRING] = {0};
    int len = 0;
    MPI_Get_library_version(lib, &len);
    for (int i = 0; i < len; ++i)
    {
        if (lib[i] == '\n')
        {
            lib[i] = '\0';
            break;
        }
    }
    std::fprintf(stderr, "[mr] procs=%d reducers=%d partition=%s combiner=%s on %s\n",
                 run.procs, run.opt.reducers, partitionName(run.opt.partition),
                 run.opt.combine ? "on" : "off", host);
    std::fprintf(stderr, "     linked against %s\n", lib);
    std::fprintf(stderr, "     thread level %s, one thread per rank, no memory shared between ranks\n",
                 threadLevelName(run.threadLevel));
    std::fflush(stderr);
}

/**
 * Opens the shared input file collectively and runs the map step over this
 * rank's byte range. Returns false on every rank if the file cannot be opened,
 * which MPI_File_open reports to all of them because the open is collective.
 *
 * The rank's own map time is stamped before MPI_File_close, which is also
 * collective and would otherwise make every rank report the slowest one.
 */
bool openAndMap(const Run& run, std::ostringstream* trace, MapResult& mapped, MPI_Offset& fileSize)
{
    MPI_File fh = MPI_FILE_NULL;
    if (MPI_File_open(MPI_COMM_WORLD, run.opt.inPath.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh)
        != MPI_SUCCESS)
    {
        if (run.rank == 0)
        {
            std::fprintf(stderr, "cannot open input file: %s\n", run.opt.inPath.c_str());
        }
        return false;
    }
    MPI_File_get_size(fh, &fileSize);
    const double start = MPI_Wtime();
    mapRange(fh, fileSize, run.rank, run.procs, run.opt.combine, trace, mapped);
    mapped.workMs = (MPI_Wtime() - start) * 1000.0;
    MPI_File_close(&fh);
    return true;
}

/** Folds every rank's map result into the hour range and totals all ranks share. Collective. */
GlobalView agree(const MapResult& mapped)
{
    GlobalView view;
    MPI_Allreduce(&mapped.minHour, &view.minHour, 1, MPI_UINT32_T, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&mapped.maxHour, &view.maxHour, 1, MPI_UINT32_T, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&mapped.records, &view.records, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&mapped.cars, &view.cars, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
    return view;
}

/** Rank 0 prints one text block per rank, in rank order, under a heading. Collective. */
void printBlocks(const Run& run, const std::string& heading, const std::string& mine)
{
    const std::vector<std::string> all = gatherText(mine, run.rank, run.procs);
    if (run.rank != 0)
    {
        return;
    }
    std::cout << "== " << heading << " ==\n";
    for (const std::string& s : all)
    {
        std::cout << s;
    }
    std::cout << "\n";
    std::cout.flush();
}

/** The lines this rank mapped and every entry it emits, with the reducer each one is bound for. */
std::string mapTraceText(const Run& run, const MapResult& mapped, const Partitioner& part,
                         const std::string& lines)
{
    std::ostringstream text;
    text << "rank " << run.rank << " maps bytes [" << mapped.start << ", " << mapped.end << "): "
         << mapped.records << " lines -> " << mapped.entries.size()
         << (run.opt.combine ? " combined keys" : " records, combiner off") << "\n"
         << lines;

    std::vector<KeyTotal> sorted = mapped.entries;
    std::sort(sorted.begin(), sorted.end(), [](const KeyTotal& a, const KeyTotal& b)
    {
        return packKey(a.hour, a.light) < packKey(b.hour, b.light);
    });
    for (const KeyTotal& e : sorted)
    {
        text << "    emit  " << traffic::formatHour(e.hour) << "  " << lightLabel(e.light)
             << "  " << std::setw(6) << e.cars << "  -> reducer " << part.destination(e) << "\n";
    }
    return text.str();
}

/** Which reducer owns each hour. Identical on every rank, so only rank 0 prints it. */
std::string planTraceText(const Partitioner& part, const GlobalView& view)
{
    std::ostringstream text;
    if (part.kind == Partition::Light)
    {
        text << "    keys go to reducer hash(hour, light) mod " << part.reducers
             << ", so one hour can span every reducer\n";
        return text.str();
    }
    for (uint32_t h = view.minHour; h <= view.maxHour; ++h)
    {
        text << "    " << traffic::formatHour(h);
        if (part.kind == Partition::Hybrid && part.ownerOfHour[h - part.minHour] == kSplitHour)
        {
            text << "  too heavy for one reducer, split by hash(hour, light) across all "
                 << part.reducers << "\n";
        }
        else
        {
            text << "  owned by reducer " << part.destination(KeyTotal{h, 0, 0}) << "\n";
        }
    }
    return text.str();
}

/**
 * Every key this reducer received, with the partial each sending rank
 * contributed and their sum. The receive displacements say which rank sent
 * which entry.
 */
std::string reduceTraceText(const Run& run, const ShuffleResult& shuffled)
{
    std::ostringstream text;
    if (run.rank >= run.opt.reducers)
    {
        text << "rank " << run.rank << " maps only and receives nothing\n";
        return text.str();
    }

    const std::vector<KeyTotal>& got = shuffled.received;
    std::vector<int> source(got.size());
    for (int r = 0; r < run.procs; ++r)
    {
        const size_t from = static_cast<size_t>(shuffled.rdispls[r]);
        const size_t to = from + static_cast<size_t>(shuffled.recvCounts[r]);
        for (size_t i = from; i < to; ++i)
        {
            source[i] = r;
        }
    }

    std::vector<size_t> order(got.size());
    for (size_t i = 0; i < order.size(); ++i)
    {
        order[i] = i;
    }
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b)
    {
        return packKey(got[a].hour, got[a].light) < packKey(got[b].hour, got[b].light);
    });

    text << "rank " << run.rank << " reduces " << got.size() << " partials\n";
    size_t i = 0;
    while (i < order.size())
    {
        const KeyTotal& key = got[order[i]];
        const uint64_t packed = packKey(key.hour, key.light);
        std::ostringstream parts;
        uint64_t sum = 0;
        for (bool first = true; i < order.size() && packKey(got[order[i]].hour, got[order[i]].light) == packed;
             first = false, ++i)
        {
            parts << (first ? "" : " + ") << got[order[i]].cars << " (rank " << source[order[i]] << ")";
            sum += got[order[i]].cars;
        }
        text << "    " << traffic::formatHour(key.hour) << "  " << lightLabel(key.light) << "  "
             << parts.str() << " = " << sum << "\n";
    }
    return text.str();
}

/** Gathers every rank's statistics, byte range and shuffle row onto rank 0. Collective. */
Collected collect(const Run& run, const RankStats& mine, const MapResult& mapped,
                  const ShuffleResult& shuffled)
{
    Collected c;
    const size_t n = run.rank == 0 ? static_cast<size_t>(run.procs) : 0;
    c.stats.resize(n);
    c.starts.resize(n);
    c.ends.resize(n);

    MPI_Gather(&mine, static_cast<int>(sizeof(RankStats)), MPI_BYTE, c.stats.data(),
               static_cast<int>(sizeof(RankStats)), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Gather(&mapped.start, 1, MPI_OFFSET, c.starts.data(), 1, MPI_OFFSET, 0, MPI_COMM_WORLD);
    MPI_Gather(&mapped.end, 1, MPI_OFFSET, c.ends.data(), 1, MPI_OFFSET, 0, MPI_COMM_WORLD);

    if (run.opt.matrix || run.opt.trace)
    {
        c.matrix.resize(n * n);
        MPI_Gather(shuffled.sendCounts.data(), run.procs, MPI_COUNT, c.matrix.data(), run.procs,
                   MPI_COUNT, 0, MPI_COMM_WORLD);
    }
    return c;
}

/** One row per rank. The byte range column is as wide as the widest range, so rows stay aligned on files of any size. */
void printRankTable(const Collected& c, int reducers)
{
    std::vector<std::string> ranges;
    int width = static_cast<int>(std::strlen("bytes"));
    for (size_t r = 0; r < c.stats.size(); ++r)
    {
        ranges.push_back("[" + std::to_string(c.starts[r]) + ", " + std::to_string(c.ends[r]) + ")");
        width = std::max(width, static_cast<int>(ranges.back().size()));
    }

    std::printf("\nrank  %-*s %10s %10s %10s %10s %10s %10s %10s %10s\n", width,
                "bytes", "records", "mapped", "->local", "->remote", "received", "keys",
                "map ms", "reduce ms");
    for (size_t r = 0; r < c.stats.size(); ++r)
    {
        const RankStats& s = c.stats[r];
        std::printf("%4zu  %-*s %10llu %10llu %10llu %10llu %10llu %10llu %10.1f %10.1f%s\n",
                    r, width, ranges[r].c_str(),
                    static_cast<unsigned long long>(s.counter[RankStats::Records]),
                    static_cast<unsigned long long>(s.counter[RankStats::Mapped]),
                    static_cast<unsigned long long>(s.counter[RankStats::SentLocal]),
                    static_cast<unsigned long long>(s.counter[RankStats::SentRemote]),
                    static_cast<unsigned long long>(s.counter[RankStats::Received]),
                    static_cast<unsigned long long>(s.counter[RankStats::Keys]),
                    s.timer[RankStats::MapMs], s.timer[RankStats::ReduceMs],
                    static_cast<int>(r) < reducers ? "" : "   map only");
    }
}

void printMatrix(const std::vector<MPI_Count>& rows, int procs)
{
    std::printf("\nshuffle matrix, entries sent from row rank to column rank\n      ");
    for (int c = 0; c < procs; ++c)
    {
        std::printf("%9d", c);
    }
    std::printf("\n");
    for (int r = 0; r < procs; ++r)
    {
        std::printf("%4d  ", r);
        for (int c = 0; c < procs; ++c)
        {
            std::printf("%9lld", static_cast<long long>(rows[static_cast<size_t>(r) * procs + c]));
        }
        std::printf("\n");
    }
}

/** Rank 0 only. Run facts, the per rank table, shuffle cost, balance, phase timings and the CSV line. */
void printSummary(const Run& run, const Partitioner& part, MPI_Offset fileSize,
                  const GlobalView& view, const Collected& c, const Phases& ph, uint64_t checksum)
{
    const Options& opt = run.opt;

    uint64_t totalMapped = 0;
    uint64_t totalRemote = 0;
    uint64_t totalKeys = 0;
    std::vector<double> received;
    std::vector<double> reduceMs;
    for (int r = 0; r < run.procs; ++r)
    {
        totalMapped += c.stats[r].counter[RankStats::Mapped];
        totalRemote += c.stats[r].counter[RankStats::SentRemote];
        totalKeys += c.stats[r].counter[RankStats::Keys];
        if (r < opt.reducers)
        {
            received.push_back(static_cast<double>(c.stats[r].counter[RankStats::Received]));
            reduceMs.push_back(c.stats[r].timer[RankStats::ReduceMs]);
        }
    }
    const double shuffleMiB = static_cast<double>(totalMapped) * sizeof(KeyTotal) / (1024.0 * 1024.0);
    const double remotePct = totalMapped ? 100.0 * static_cast<double>(totalRemote) /
                                           static_cast<double>(totalMapped) : 0.0;
    const double entryImbalance = imbalance(received);
    const double timeImbalance = imbalance(reduceMs);

    std::printf("\nprogram     traffic_mr\n");
    std::printf("input       %s (%.2f MiB)\n", opt.inPath.c_str(),
                static_cast<double>(fileSize) / (1024.0 * 1024.0));
    std::printf("processes   %d map, %d reduce\n", run.procs, opt.reducers);
    if (opt.partition == Partition::Hybrid)
    {
        std::printf("partition   hybrid, %zu hour%s split by light, the rest weighted\n",
                    part.splitHours, part.splitHours == 1 ? "" : "s");
    }
    else
    {
        std::printf("partition   %s\n", partitionName(opt.partition));
    }
    std::printf("combiner    %s\n", opt.combine ? "on" : "off");
    std::printf("records     %llu\n", static_cast<unsigned long long>(view.records));
    std::printf("cars        %llu\n", static_cast<unsigned long long>(view.cars));
    std::printf("keys        %llu\n", static_cast<unsigned long long>(totalKeys));
    std::printf("hours       %u  (%s .. %s)\n", view.maxHour - view.minHour + 1,
                traffic::formatHour(view.minHour).c_str(), traffic::formatHour(view.maxHour).c_str());

    printRankTable(c, opt.reducers);
    if (!c.matrix.empty())
    {
        printMatrix(c.matrix, run.procs);
    }

    std::printf("\nshuffle     %llu entries, %.2f MiB, %.1f%% crossed to another rank\n",
                static_cast<unsigned long long>(totalMapped), shuffleMiB, remotePct);
    std::printf("imbalance   entries received max/mean %.3f   reduce time max/mean %.3f\n",
                entryImbalance, timeImbalance);
    std::printf("phases      map %.2f  plan %.2f  shuffle %.2f  reduce %.2f  gather %.2f  (ms)\n",
                ph.map, ph.plan, ph.shuffle, ph.reduce, ph.gather);
    std::printf("total       %.2f ms\n", ph.total());
    if (opt.verify)
    {
        std::printf("checksum    %llu\n", static_cast<unsigned long long>(checksum));
    }
    else
    {
        std::printf("checksum    not computed, pass --verify\n");
    }

    if (opt.csv)
    {
        std::printf("CSV,mr,%d,%d,%s,%s,%llu,%llu,%llu,%llu,%.4f,%.4f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%llu\n",
                    run.procs, opt.reducers, partitionName(opt.partition), opt.combine ? "on" : "off",
                    static_cast<unsigned long long>(view.records),
                    static_cast<unsigned long long>(totalKeys),
                    static_cast<unsigned long long>(totalMapped),
                    static_cast<unsigned long long>(totalRemote),
                    entryImbalance, timeImbalance,
                    ph.map, ph.plan, ph.shuffle, ph.reduce, ph.gather, ph.total(),
                    static_cast<unsigned long long>(checksum));
    }
    std::fflush(stdout);
}

void finish(Run& run)
{
    if (run.keyTotalType != MPI_DATATYPE_NULL)
    {
        MPI_Type_free(&run.keyTotalType);
    }
    MPI_Finalize();
}

/**
 * Runs map, plan, shuffle, reduce and gather, then reports on rank 0.
 * Records the current phase in run.phase so an allocation failure can say
 * where the rank ran out of memory.
 */
int runPhases(Run& run)
{
    Phases ph;
    RankStats mine;
    std::ostringstream lineTrace;

    run.phase = "map";
    progress(run, "mapping " + run.opt.inPath);
    double clock = 0;
    resetClock(clock);

    MapResult mapped;
    MPI_Offset fileSize = 0;
    if (!openAndMap(run, run.opt.trace ? &lineTrace : nullptr, mapped, fileSize))
    {
        return 1;
    }
    mine.timer[RankStats::MapMs] = mapped.workMs;
    mine.counter[RankStats::Records] = mapped.records;
    mine.counter[RankStats::Mapped] = mapped.entries.size();
    ph.map = closePhase(clock);

    run.phase = "plan";
    progress(run, std::string("planning the ") + partitionName(run.opt.partition) + " partition");
    const GlobalView view = agree(mapped);
    if (view.records == 0)
    {
        if (run.rank == 0)
        {
            std::fprintf(stderr, "no records in %s\n", run.opt.inPath.c_str());
        }
        return 1;
    }
    const Partitioner part = buildPartitioner(run.opt.partition, run.opt.reducers, view.minHour,
                                              view.maxHour, mapped.entries);
    ph.plan = closePhase(clock);

    if (run.opt.trace)
    {
        printBlocks(run, "map", mapTraceText(run, mapped, part, lineTrace.str()));
        if (run.rank == 0)
        {
            std::cout << "== plan ==\n" << planTraceText(part, view) << "\n";
            std::cout.flush();
        }
        resetClock(clock);
    }

    run.phase = "shuffle";
    progress(run, "shuffling");
    ShuffleResult shuffled;
    shuffle(mapped.entries, part, run.procs, run.keyTotalType, shuffled);
    mine.timer[RankStats::ShuffleMs] = (MPI_Wtime() - clock) * 1000.0;
    for (int r = 0; r < run.procs; ++r)
    {
        mine.counter[r == run.rank ? RankStats::SentLocal : RankStats::SentRemote] +=
            static_cast<uint64_t>(shuffled.sendCounts[r]);
    }
    mine.counter[RankStats::Received] = shuffled.received.size();
    ph.shuffle = closePhase(clock);

    std::string reduceTrace;
    if (run.opt.trace)
    {
        reduceTrace = reduceTraceText(run, shuffled);
        resetClock(clock);
    }

    run.phase = "reduce";
    progress(run, "reducing");
    ReduceResult reduced;
    if (run.rank < run.opt.reducers)
    {
        reduceReceived(shuffled, run.opt.topN, reduced);
    }
    mine.timer[RankStats::ReduceMs] = (MPI_Wtime() - clock) * 1000.0;
    mine.counter[RankStats::Keys] = reduced.keys;
    mine.counter[RankStats::Candidates] = reduced.candidates.size();
    ph.reduce = closePhase(clock);

    if (run.opt.trace)
    {
        printBlocks(run, "reduce", reduceTrace);
        resetClock(clock);
    }

    run.phase = "gather";
    progress(run, "gathering");
    const std::vector<KeyTotal> candidates = gatherEntries(reduced.candidates, run.rank, run.procs,
                                                           run.keyTotalType);
    std::vector<KeyTotal> ranked;
    if (run.rank == 0)
    {
        ranked = mergeCandidates(candidates, run.opt.topN);
    }
    ph.gather = closePhase(clock);

    const Collected collected = collect(run, mine, mapped, shuffled);

    uint64_t checksum = 0;
    if (run.opt.verify)
    {
        run.phase = "verify";
        progress(run, "verifying, gathering every total to rank 0");
        std::vector<KeyTotal> local;
        local.reserve(reduced.keys);
        if (run.rank < run.opt.reducers)
        {
            forEachTotal(shuffled.received, [&local](uint32_t hour, uint32_t light, uint64_t cars)
            {
                local.push_back(KeyTotal{hour, light, cars});
            });
        }
        const std::vector<KeyTotal> totals = gatherEntries(local, run.rank, run.procs,
                                                           run.keyTotalType);
        if (run.rank == 0)
        {
            checksum = summaryChecksum(totals, view.minHour);
        }
    }

    if (run.rank == 0)
    {
        report(ranked, run.opt.topN, run.opt.printHours, run.opt.outSummary);
        printSummary(run, part, fileSize, view, collected, ph, checksum);
    }

    return 0;
}

}  // namespace

int main(int argc, char** argv)
{
    Run run;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &run.threadLevel);
    MPI_Comm_rank(MPI_COMM_WORLD, &run.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &run.procs);

    int exitCode = 0;
    if (!configure(argc, argv, run, exitCode))
    {
        finish(run);
        return exitCode;
    }
    run.keyTotalType = makeKeyTotalType();
    printBanner(run);

    try
    {
        exitCode = runPhases(run);
    }
    catch (const std::bad_alloc&)
    {
        fatal(run.rank, std::string("out of memory (std::bad_alloc) in the ") + run.phase + " phase");
    }

    finish(run);
    return exitCode;
}
