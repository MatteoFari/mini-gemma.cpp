#include "engine.h"
#include "sampler.h"
#include <algorithm>
#include <cstdio>
#include <fcntl.h>
#include <unistd.h>
#include <ostream>
#include <stdexcept>

LLMEngine::LLMEngine(const std::string &path, const Options &options)
    : options_(options), loader_(path), tokenizer_(loader_),
      model_(loader_, options.context, options.threads),
      rng_(options.seed) {
}
void LLMEngine::chat(const std::string &prompt, std::ostream &out) {
    if (prompt.find_first_not_of(" \t\r\n") == std::string::npos)
        throw std::runtime_error("Enter a non-empty prompt");
    auto start = Clock::now();
    // Start new conversations with the system turn and BOS.
    bool initial = model_.position() == 0;
    std::string system = initial ? "<|turn>system\n" + options_.system + "<turn|>\n" : "";
    std::string text = system + "<|turn>user\n" + prompt + "<turn|>\n<|turn>model\n";
    auto tokens = tokenizer_.encode(text, initial);
    auto closing = tokenizer_.encode("<turn|>\n", false);
    // Reserve prompt, reply, and closing space before changing state.
    if (model_.position() + tokens.size() + options_.max_tokens + closing.size() > size_t(model_.capacity()))
        throw std::runtime_error(
            "Not enough context for prompt and output budget; use /reset or reduce --max-tokens");
    double boot = profiler_.boot;
    profiler_ = {};
    profiler_.boot = boot;
    profiler_.prompt_tokens = int(tokens.size());
    std::span<const float> scores;
    // Fill the KV cache. Only the last prompt token needs logits.
    for (size_t i = 0; i < tokens.size(); ++i) {
        scores = model_.forward(tokens[i], i + 1 == tokens.size());
    }
    profiler_.prefill = elapsed(start);
    Clock::time_point first{};
    // Generate and stream one token at a time.
    for (int i = 0; i < options_.max_tokens; ++i) {
        int next = sample(scores, options_.temperature, options_.top_p, rng_);
        if (tokenizer_.stop(next)) break;
        out << tokenizer_.decode(next) << std::flush;
        ++profiler_.generated_tokens;
        if (i == 0) {
            profiler_.first_token = elapsed(start);
            first = Clock::now();
        } else
            profiler_.decode = elapsed(first);
        scores = model_.forward(next, i + 1 < options_.max_tokens);
    }
    // Close the model turn even when the output limit stops generation.
    for (int token : closing)
        model_.forward(token, false);
    out << '\n';
}
void LLMEngine::reset() {
    // Reset the conversation and sampling.
    model_.reset();
    rng_.seed(options_.seed);
    double boot = profiler_.boot;
    profiler_ = {};
    profiler_.boot = boot;
}
void LLMEngine::metrics(std::ostream &out) const {
    profiler_.print(out, model_.position(), model_.capacity(), model_.cache_bytes(), model_.scratch_bytes());
}
