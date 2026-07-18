#pragma once
#include <chrono>
#include <iosfwd>
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start);
struct Profiler {
    double boot = 0, prefill = 0, first_token = 0, decode = 0;
    int prompt_tokens = 0, generated_tokens = 0, reused_tokens = 0;
    void print(std::ostream &out, int position, int capacity, size_t cache, size_t scratch) const;
};
