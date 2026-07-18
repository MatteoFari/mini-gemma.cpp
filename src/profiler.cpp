#include "profiler.h"
#include <algorithm>
#include <iomanip>
#include <ostream>
double elapsed(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}
void Profiler::print(std::ostream &out, int position, int capacity, size_t cache, size_t scratch) const {
    // Decode speed excludes the first token, which has its own timer.
    out << std::fixed << std::setprecision(3) << "Boot: " << boot << " s | context: " << position << '/'
        << capacity << "\n"
        << "Prompt: " << prompt_tokens << " tokens | prefill: " << prefill << " s | reused: " << reused_tokens
        << " tokens | computed: " << prompt_tokens - reused_tokens << " tokens\n"
        << "Generated: " << generated_tokens << " tokens | first token: " << first_token << " s\n"
        << "Decode: " << (decode > 0 ? std::max(0, generated_tokens - 1) / decode : 0) << " tokens/s\n"
        << "KV cache: " << cache / 1048576.0 << " MiB | graph scratch: " << scratch / 1048576.0 << " MiB\n";
}
