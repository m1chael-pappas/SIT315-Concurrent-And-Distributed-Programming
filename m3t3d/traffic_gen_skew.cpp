/*
 * SIT315 Module 3 - Task M3.T3D - Traffic Control Simulator, skewed data generator
 * Michael Pappas
 *
 * Writes a traffic file in the m2t3d format whose keys are unevenly spread
 * across hours, so the partitioning strategies in traffic_mr have something to
 * disagree about. Car counts come from the same model as m2t3d's generator.
 *
 *   rollout   the network grows from --rollout of --lights to all of them over
 *             the period, so later hours carry more keys than earlier ones
 *   diurnal   only a rush hour weighted share of the installed lights report in
 *             any hour, so keys per hour follow a 24 hour cycle
 *   incident  one hour adds enough temporary counters that it holds
 *             --incident-share of every key in the file
 *
 * Rows stay in m2t3d order (hour, then 5 minute slot, then light id) so
 * ../m2t3d/traffic_sim can read the file when given --lights.
 *
 * Build:
 *   g++ -std=c++17 -O2 -Wall -Wextra -o traffic_gen_skew traffic_gen_skew.cpp
 *
 * Run:
 *   ./traffic_gen_skew --out data/skew_medium.csv --lights 1000 --hours 1680
 */

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "traffic_common.h"

namespace
{

struct Options
{
    std::string outPath;
    uint32_t lights = 0;
    uint32_t hours = 0;
    uint64_t seed = 42;
    double rollout = 0.1;
    double incidentShare = 0.25;
    long long incidentHour = -1;
};

void usage()
{
    std::cout <<
        "SIT315 M3.T3D skewed traffic data generator\n"
        "\n"
        "  traffic_gen_skew --out FILE --lights N --hours H [--seed S]\n"
        "                   [--rollout F] [--incident-share X] [--incident-hour K]\n"
        "\n"
        "  --rollout         share of lights installed in the first hour (default 0.1)\n"
        "  --incident-share  share of all keys that fall in the incident hour, 0 for none (default 0.25)\n"
        "  --incident-hour   hour index of the incident (default hours / 2)\n";
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
        if (flag == "--out")
        {
            if (!next(value)) return false;
            opt.outPath = value;
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
        else if (flag == "--rollout")
        {
            if (!next(value)) return false;
            opt.rollout = std::stod(value);
        }
        else if (flag == "--incident-share")
        {
            if (!next(value)) return false;
            opt.incidentShare = std::stod(value);
        }
        else if (flag == "--incident-hour")
        {
            if (!next(value)) return false;
            opt.incidentHour = std::stoll(value);
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

    if (opt.outPath.empty() || opt.lights == 0 || opt.hours == 0)
    {
        error = "--out, --lights and --hours are required";
        return false;
    }
    if (opt.rollout <= 0 || opt.rollout > 1)
    {
        error = "--rollout must be in (0, 1]";
        return false;
    }
    if (opt.incidentShare < 0 || opt.incidentShare >= 1)
    {
        error = "--incident-share must be in [0, 1)";
        return false;
    }
    if (opt.incidentHour < 0)
    {
        opt.incidentHour = opt.hours / 2;
    }
    if (opt.incidentHour >= static_cast<long long>(opt.hours))
    {
        error = "--incident-hour must be below --hours";
        return false;
    }
    return true;
}

/** Lights reporting in hour h: the installed share of the network, scaled by that hour's rush weight. */
std::vector<uint32_t> activeLightsPerHour(const Options& opt, long long baseHour)
{
    std::vector<uint32_t> active(opt.hours);
    for (uint32_t h = 0; h < opt.hours; ++h)
    {
        const double progress = opt.hours > 1 ? static_cast<double>(h) / (opt.hours - 1) : 1.0;
        const double installed = opt.rollout + (1.0 - opt.rollout) * progress;
        const int weight = traffic::rushWeight[(baseHour + h) % 24];
        const double reporting = installed * weight / 100.0;
        active[h] = std::max<uint32_t>(1, static_cast<uint32_t>(std::lround(opt.lights * reporting)));
    }
    return active;
}

/** Extra temporary counters needed so the incident hour holds the requested share of every key. */
uint32_t incidentExtraLights(const Options& opt, const std::vector<uint32_t>& active)
{
    if (opt.incidentShare <= 0)
    {
        return 0;
    }
    uint64_t others = 0;
    for (uint32_t h = 0; h < opt.hours; ++h)
    {
        if (h != opt.incidentHour)
        {
            others += active[h];
        }
    }
    const double wantInHour = opt.incidentShare * static_cast<double>(others) / (1.0 - opt.incidentShare);
    const double extra = wantInHour - active[static_cast<size_t>(opt.incidentHour)];
    return extra > 0 ? static_cast<uint32_t>(std::ceil(extra)) : 0;
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

    const long long baseHour = traffic::daysFromCivil(2026, 8, 1) * 24;
    const std::vector<uint32_t> active = activeLightsPerHour(opt, baseHour);
    const uint32_t extra = incidentExtraLights(opt, active);
    const uint32_t maxLights = std::max(opt.lights, opt.lights + extra);

    std::vector<uint32_t> baseRate(maxLights);
    for (uint32_t l = 0; l < maxLights; ++l)
    {
        baseRate[l] = traffic::baseRateFor(l);
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

    const auto start = traffic::Clock::now();
    uint64_t lines = 0;
    uint64_t keys = 0;
    uint64_t heaviest = 0;
    uint32_t heaviestHour = 0;

    for (uint32_t h = 0; h < opt.hours; ++h)
    {
        const long long absHour = baseHour + h;
        const int hourOfDay = static_cast<int>(absHour % 24);
        const int weight = traffic::rushWeight[hourOfDay];
        const bool incident = h == opt.incidentHour;

        int y = 0;
        unsigned mo = 0;
        unsigned da = 0;
        traffic::civilFromDays(absHour / 24, y, mo, da);

        std::vector<uint32_t> ids;
        ids.reserve(active[h] + (incident ? extra : 0));
        for (uint32_t l = 0; l < active[h]; ++l)
        {
            ids.push_back(l);
        }
        if (incident)
        {
            for (uint32_t l = 0; l < extra; ++l)
            {
                ids.push_back(opt.lights + l);
            }
        }
        keys += ids.size();
        if (ids.size() > heaviest)
        {
            heaviest = ids.size();
            heaviestHour = h;
        }

        for (int slot = 0; slot < 12; ++slot)
        {
            char stamp[40];
            std::snprintf(stamp, sizeof(stamp), "%04d-%02u-%02u %02d:%02d",
                          y, mo, da, hourOfDay, slot * 5);
            const size_t stampLen = std::strlen(stamp);

            for (uint32_t l : ids)
            {
                const long long mean = (static_cast<long long>(baseRate[l]) * weight) / 100;
                const long long spread = mean / 3 + 4;
                long long value = mean + static_cast<long long>(rng() % static_cast<uint64_t>(2 * spread + 1)) - spread;
                if (value < 0)
                {
                    value = 0;
                }

                buf.insert(buf.end(), stamp, stamp + stampLen);
                buf.push_back(',');
                buf.push_back('T');
                buf.push_back('L');
                if (l < 1000) buf.push_back('0');
                if (l < 100)  buf.push_back('0');
                if (l < 10)   buf.push_back('0');
                traffic::appendUInt(buf, l);
                buf.push_back(',');
                traffic::appendUInt(buf, static_cast<uint64_t>(value));
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
              << "  lights            : " << opt.lights << " installed at full rollout, "
              << extra << " temporary in the incident hour\n"
              << "  hours             : " << opt.hours << "\n"
              << "  rows              : " << lines << "\n"
              << "  keys              : " << keys << "  (distinct hour, light pairs)\n"
              << "  keys per hour     : min " << *std::min_element(active.begin(), active.end())
              << ", mean " << keys / opt.hours << ", max " << heaviest << " at "
              << traffic::formatHour(baseHour + heaviestHour) << "\n"
              << "  heaviest hour     : " << (100.0 * static_cast<double>(heaviest) / static_cast<double>(keys))
              << "% of all keys\n"
              << "  size              : " << (bytes / (1024.0 * 1024.0)) << " MiB\n"
              << "  elapsed           : " << traffic::msSince(start) << " ms\n"
              << "\nCheck it against m2t3d with:\n"
              << "  ../m2t3d/traffic_sim seq --in " << opt.outPath << " --lights " << maxLights << "\n";
    return 0;
}
