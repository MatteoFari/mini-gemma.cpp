#include "prefix_cache.h"
#include "profiler.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}
int main(int argc, char **argv) try {
    if (argc != 2) throw std::runtime_error("Expected model path");
    ModelLoader loader(argv[1]);
    Model model(loader, 768, 4);
    PrefixCache cache(model, 128 * 1048576);
    std::vector<int> tokens(541, 496); // Same BOS + repeated ' a' probe as the independent oracle.
    tokens[0] = 2;
    auto finish = [&](std::span<const int> input) {
        std::span<const float> scores;
        for (size_t i = model.position(); i < input.size(); ++i)
            scores = model.forward(input[i], i + 1 == input.size());
        return std::vector<float>(scores.begin(), scores.end());
    };
    auto start = Clock::now();
    for (size_t i = 0; i < 539; ++i) {
        model.forward(tokens[i], false);
        if (i == 63 || i == 127 || i == 538) cache.remember(std::span(tokens).first(i + 1));
    }
    auto expected = finish(tokens);
    double cold = elapsed(start);
    require(expected.size() == 262144 &&
                std::all_of(expected.begin(), expected.end(), [](float x) { return std::isfinite(x); }),
            "Invalid cold logits");
    for (int i = 0; i < 16; ++i)
        model.forward(9259, false); // Contaminate live ring and full caches.
    model.reset();
    start = Clock::now();
    require(cache.restore(tokens) == 539, "Longest prefix after ring wrap was not selected");
    require(finish(tokens) == expected, "Restored ring logits differ from cold logits");
    double warm = elapsed(start);
    // A different question after a shared prefix must compute its own continuation.
    std::vector<int> branch(tokens.begin(), tokens.begin() + 150);
    branch[140] = 9259;
    model.reset();
    auto branch_scores = finish(branch);
    model.reset();
    require(cache.restore(branch) == 128, "Shared-prefix branch chose the wrong checkpoint");
    require(finish(branch) == branch_scores, "Shared-prefix branch logits changed");
    auto miss = branch;
    miss[0] = 1;
    model.reset();
    require(cache.restore(miss) == 0 && model.position() == 0, "Mismatching tokens restored state");
    // Two snapshots fit. After touching A, inserting C must evict B.
    PrefixCache lru(model, 3 * 1048576);
    auto a = std::vector<int>(tokens.begin(), tokens.begin() + 33), b = a, c = a;
    b[1] = 9259;
    c[1] = 1;
    auto save = [&](const std::vector<int> &input) {
        model.reset();
        for (size_t i = 0; i < 32; ++i)
            model.forward(input[i], false);
        lru.remember(std::span(input).first(32));
    };
    save(a);
    size_t one_entry = lru.bytes();
    lru.remember(std::span(a).first(32));
    require(lru.bytes() == one_entry, "Duplicate checkpoint consumed memory");
    save(b);
    model.reset();
    require(lru.restore(a) == 32, "LRU touch failed");
    save(c);
    require(lru.evictions() == 1 && lru.bytes() <= lru.budget(), "Memory budget eviction failed");
    model.reset();
    require(lru.restore(b) == 0, "Least recently used checkpoint survived eviction");
    require(lru.restore(a) == 32, "Recently used checkpoint was evicted");
    model.reset();
    require(lru.restore(c) == 32, "New checkpoint was not retained");
    model.reset();
    require(lru.restore(std::span(c).first(32)) == 0, "Restore must leave a token for logits");
    cache.restore(tokens);
    size_t retained = lru.bytes();
    lru.remember(std::span(tokens).first(model.position()));
    require(lru.bytes() == retained && lru.evictions() == 1, "Oversized entry evicted useful state");
    bool rejected = false;
    try {
        cache.restore(tokens);
    } catch (const std::runtime_error &) {
        rejected = true;
    }
    require(rejected && model.position() == 539, "Active conversation was overwritten");
    Model other(loader, 768, 4);
    rejected = false;
    try {
        other.restore(model.snapshot());
    } catch (const std::runtime_error &) {
        rejected = true;
    }
    require(rejected && other.position() == 0, "Foreign snapshot was accepted");
    std::cout << "Prefix tests passed: exact 262144-score equality for ring wrap and branching; "
              << "misses, LRU, budget, duplicate and ownership guards.\n"
              << "Diagnostic prefill: cold=" << cold << " s, reused=" << warm << " s, 539/541 tokens.\n";
    return 0;
} catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
}
