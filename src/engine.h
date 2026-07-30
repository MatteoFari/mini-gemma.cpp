#pragma once
#include "model.h"
#include "tokenizer.h"
#include "profiler.h"
#include "prefix_cache.h"
#include <random>

struct Options {
    int context = 2048, threads = 4, max_tokens = 128;
    float temperature = 0, top_p = 0.9f;
    unsigned seed = 42;
    size_t prefix_cache_mib = 0;
    std::string system;
};
class LLMEngine {
    Options options_;
    ModelLoader loader_;
    Tokenizer tokenizer_;
    Model model_;
    PrefixCache prefixes_;
    std::mt19937 rng_;
    Profiler profiler_;

  public:
    LLMEngine(const std::string &path, const Options &options);
    void chat(const std::string &prompt, std::ostream &out);
    void reset();
    void metrics(std::ostream &out) const;
    void set_boot(double seconds) { profiler_.boot = seconds; }
    const Tokenizer &tokenizer() const { return tokenizer_; }
    void logits(const std::string &text, const std::string &path);
};
