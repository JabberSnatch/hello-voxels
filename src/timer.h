#pragma once

#include <chrono>
#include <iostream>

struct Mute
{
    static void log(const char*, float) {}
};
struct Active
{
    static void log(const char* label, float time)
    {
        std::cout << label << " " << time << " ms" << std::endl;
    }
};
struct Verbose : public Mute
{};

struct Info : public Active
{};

template <typename Verbosity = Verbose>
struct Timer
{
    using StdClock = std::chrono::high_resolution_clock;

    Timer(const char* label) :
        label_(label),
        start(StdClock::now()) {}
    ~Timer()
    {
        StdClock::time_point end = StdClock::now();
        float dt = static_cast<float>(std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());
        Verbosity::log(label_, dt * 0.001);
    }

    const char* label_;
    StdClock::time_point start;
};
