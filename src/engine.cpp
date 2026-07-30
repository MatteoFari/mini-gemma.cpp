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
      prefixes_(model_, options.prefix_cache_mib * 1048576), rng_(options.seed) {
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
    size_t system_boundary = 0;
    if (initial && prefixes_.enabled()) {
        auto fixed = tokenizer_.encode(system, true);
        // Text boundaries can merge, so compare token IDs before caching.
        system_boundary =
            std::mismatch(tokens.begin(), tokens.end(), fixed.begin(), fixed.end()).first - tokens.begin();
    }
    size_t reused = initial && prefixes_.enabled() ? prefixes_.restore(tokens) : 0;
    profiler_.reused_tokens = int(reused);
    std::span<const float> scores;
    // Fill the KV cache. Only the last prompt token needs logits.
    for (size_t i = reused; i < tokens.size(); ++i) {
        scores = model_.forward(tokens[i], i + 1 == tokens.size());
        // Save checkpoints for questions that share an earlier prefix.
        if (initial && prefixes_.enabled() && i + 1 < tokens.size() &&
            ((i + 1) % 64 == 0 || i + 1 == system_boundary || i + 2 == tokens.size()))
            prefixes_.remember(std::span(tokens).first(i + 1));
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
    // Reset the conversation and sampling, keeping prefix checkpoints.
    model_.reset();
    rng_.seed(options_.seed);
    double boot = profiler_.boot;
    profiler_ = {};
    profiler_.boot = boot;
}
void LLMEngine::metrics(std::ostream &out) const {
    profiler_.print(out, model_.position(), model_.capacity(), model_.cache_bytes(), model_.scratch_bytes());
    if (prefixes_.enabled())
        out << "Prefix cache: " << prefixes_.bytes() / 1048576.0 << '/' << prefixes_.budget() / 1048576.0
            << " MiB | evictions: " << prefixes_.evictions() << " total\n";
}
void LLMEngine::logits(const std::string &text, const std::string &path) {
    auto tokens = tokenizer_.encode(text);
    if (tokens.empty() || tokens.size() > size_t(model_.capacity()))
        throw std::runtime_error("Invalid probe length");
    reset();
    std::span<const float> values;
    for (size_t i = 0; i < tokens.size(); ++i)
        values = model_.forward(tokens[i], i + 1 == tokens.size());
    // Check file identity before truncating, to protect the mapped weights.
    int descriptor = open(path.c_str(), O_WRONLY | O_CREAT | O_CLOEXEC | O_NONBLOCK, 0600);
    if (descriptor < 0) throw std::runtime_error("Cannot open logits output");
    std::unique_ptr<FILE, decltype(&std::fclose)> file(fdopen(descriptor, "wb"), std::fclose);
    if (!file) {
        close(descriptor);
        throw std::runtime_error("Cannot create logits stream");
    }
    if (loader_.is_model_file(descriptor)) throw std::runtime_error("Logits output aliases the model file");
    if (ftruncate(descriptor, 0) != 0 ||
        std::fwrite(values.data(), sizeof(float), values.size(), file.get()) != values.size() ||
        std::fflush(file.get()) != 0)
        throw std::runtime_error("Cannot write logits");
}
