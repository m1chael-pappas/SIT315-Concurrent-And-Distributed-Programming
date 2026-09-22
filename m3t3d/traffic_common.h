/*
 * SIT315 Module 3 - Task M3.T3D - Traffic Control Simulator, shared pieces
 * Michael Pappas
 *
 * Copied from m2t3d/traffic_sim.cpp so both programs in this folder parse,
 * date-stamp, rank and generate data exactly the way M2.T3D did. The function
 * bodies are unchanged apart from being marked inline. Matching the M2 code
 * byte for byte is what lets an --out-summary file from traffic_mr be compared
 * with cmp against one from ../m2t3d/traffic_sim.
 *
 * rushWeight and baseRateFor are lifted out of m2t3d's runGenerate so the
 * skewed generator draws car counts from the same model.
 */

#ifndef TRAFFIC_COMMON_H
#define TRAFFIC_COMMON_H

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace traffic
{

using Clock = std::chrono::steady_clock;

inline double msSince(const Clock::time_point& start)
{
    const auto now = Clock::now();
    return std::chrono::duration<double, std::milli>(now - start).count();
}

// ---------------------------------------------------------------------------
// Calendar helpers (Howard Hinnant's civil date algorithms)
// ---------------------------------------------------------------------------

inline long long daysFromCivil(int y, unsigned m, unsigned d)
{
    y -= m <= 2;
    const int era = (y >= 0 ? y : y - 399) / 400;
    const unsigned yoe = static_cast<unsigned>(y - era * 400);
    const unsigned doy = (153u * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
    const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return static_cast<long long>(era) * 146097 + static_cast<long long>(doe) - 719468;
}

inline void civilFromDays(long long z, int& y, unsigned& m, unsigned& d)
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

inline std::string formatHour(long long absHour)
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
inline bool parseLine(const char* s, size_t n, long long& absHour, uint32_t& light, uint32_t& cars)
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
// Ranking
// ---------------------------------------------------------------------------

struct LightTotal
{
    uint32_t light;
    uint64_t cars;
};

inline bool busierFirst(const LightTotal& a, const LightTotal& b)
{
    if (a.cars != b.cars)
    {
        return a.cars > b.cars;
    }
    return a.light < b.light;  // stable tie break so both modes agree
}

// ---------------------------------------------------------------------------
// Generation
// ---------------------------------------------------------------------------

inline void appendUInt(std::vector<char>& buf, uint64_t v)
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

// Rush hour weights push the morning and evening peaks up so the top N list
// changes across the day instead of being flat noise.
static const int rushWeight[24] =
{
    30, 22, 18, 16, 20, 35, 60, 90, 100, 85, 70, 68,
    72, 70, 68, 75, 92, 100, 88, 70, 58, 48, 40, 34
};

// Each light gets a fixed base rate, and one light in every 89 is a busy
// intersection.
inline uint32_t baseRateFor(uint32_t l)
{
    uint32_t base = 18 + (l % 7) * 6;
    if (l % 89 == 0)
    {
        base *= 3;
    }
    return base;
}

}  // namespace traffic

#endif
