#include "engine.h"
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>

namespace {
void help() {
    std::cout << "Usage: inference_engine MODEL.gguf [--prompt TEXT] [--threads N] [--ctx N]\n"
              << "  --max-tokens N --temp F --top-p F --seed N --system TEXT\n"
              << "  --prefix-cache-mib N enables reusable prompt checkpoints (default 0, max 1024)\n"
              << "  --tokens TEXT prints token IDs. --logits FILE --prompt TEXT writes raw F32 scores\n"
              << "Interactive commands: /metrics /reset /help /exit\n";
}
int integer(const std::string &value) {
    size_t used = 0;
    int n = std::stoi(value, &used);
    if (used != value.size()) throw std::runtime_error("Invalid integer: " + value);
    return n;
}
float number(const std::string &value) {
    size_t used = 0;
    float n = std::stof(value, &used);
    if (used != value.size() || !std::isfinite(n)) throw std::runtime_error("Invalid number: " + value);
    return n;
}
} // namespace
int main(int argc, char **argv) try {
    if (argc == 2 && std::string(argv[1]) == "--help") {
        help();
        return 0;
    }
    if (argc < 2) {
        help();
        return 1;
    }
    // Read generation settings before loading the model.
    Options options;
    std::string prompt, token_text, dump;
    bool one_shot = false, tokenize = false;
    for (int i = 2; i < argc; ++i) {
        std::string flag = argv[i];
        if (++i >= argc) throw std::runtime_error("Missing value for " + flag);
        std::string value = argv[i];
        if (flag == "--prompt") {
            prompt = value;
            one_shot = true;
        } else if (flag == "--tokens") {
            token_text = value;
            tokenize = true;
        } else if (flag == "--logits")
            dump = value;
        else if (flag == "--threads")
            options.threads = integer(value);
        else if (flag == "--ctx")
            options.context = integer(value);
        else if (flag == "--max-tokens")
            options.max_tokens = integer(value);
        else if (flag == "--prefix-cache-mib") {
            int mib = integer(value);
            if (mib < 0 || mib > 1024) throw std::runtime_error("Prefix cache budget must be 0..1024 MiB");
            options.prefix_cache_mib = size_t(mib);
        } else if (flag == "--seed") {
            int seed = integer(value);
            if (seed < 0) throw std::runtime_error("Negative seed");
            options.seed = unsigned(seed);
        } else if (flag == "--temp")
            options.temperature = number(value);
        else if (flag == "--top-p")
            options.top_p = number(value);
        else if (flag == "--system")
            options.system = value;
        else
            throw std::runtime_error("Unknown option: " + flag);
    }
    if (options.max_tokens < 1 || options.max_tokens > 8192 || options.temperature < 0 ||
        options.top_p <= 0 || options.top_p > 1)
        throw std::runtime_error("Invalid generation settings");
    if (!dump.empty() && !one_shot) throw std::runtime_error("--logits requires --prompt");
    // Token inspection needs the vocabulary but no inference graph.
    if (tokenize) {
        ModelLoader loader(argv[1]);
        Tokenizer tokenizer(loader);
        for (int id : tokenizer.encode(token_text))
            std::cout << id << ' ';
        std::cout << '\n';
        return 0;
    }
    auto start = Clock::now();
    LLMEngine engine(argv[1], options);
    engine.set_boot(elapsed(start));
    std::cerr << "Model ready on CPU in " << elapsed(start) << " s\n";
    if (!dump.empty()) {
        engine.logits(prompt, dump);
        return 0;
    }
    if (one_shot) {
        engine.chat(prompt, std::cout);
        engine.metrics(std::cerr);
        return 0;
    }
    help();
    for (std::string input; std::cout << "User > " && std::getline(std::cin, input);) {
        if (input == "/exit") break;
        if (input == "/help") {
            help();
            continue;
        }
        if (input == "/metrics") {
            engine.metrics(std::cout);
            continue;
        }
        if (input == "/reset") {
            engine.reset();
            std::cout << "Context cleared.\n";
            continue;
        }
        if (input.empty()) continue;
        try {
            std::cout << "AI > ";
            engine.chat(input, std::cout);
        } catch (const std::exception &error) {
            // Keep the chat open after a failed request.
            std::cout << "Error: " << error.what() << '\n';
        }
    }
    return 0;
} catch (const std::exception &error) {
    std::cerr << "Error: " << error.what() << '\n';
    return 1;
}
