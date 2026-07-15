#include "sampler.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

int sample(std::span<const float> logits, float temperature, float top_p, std::mt19937 &rng) {
    if (logits.empty() || !std::isfinite(temperature) || temperature < 0 || !std::isfinite(top_p) ||
        top_p <= 0 || top_p > 1)
        throw std::runtime_error("Invalid sampling parameters");
    for (float x : logits)
        if (!std::isfinite(x)) throw std::runtime_error("Non-finite model logits");
    int best = int(std::max_element(logits.begin(), logits.end()) - logits.begin());
    // Zero temperature selects the highest score.
    if (temperature == 0) return best;
    std::vector<int> order(logits.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return logits[a] > logits[b] || (logits[a] == logits[b] && a < b); });
    // Subtract the largest logit to keep exponentials bounded.
    std::vector<double> weights;
    double total = 0;
    for (int id : order) {
        weights.push_back(std::exp(double(logits[id] - logits[best]) / temperature));
        total += weights.back();
    }
    // Keep the most likely tokens until their mass reaches top-p.
    double cumulative = 0;
    size_t count = 0;
    do {
        cumulative += weights[count++];
    } while (count < weights.size() && cumulative < top_p * total);
    std::discrete_distribution<size_t> distribution(weights.begin(), weights.begin() + count);
    return order[distribution(rng)];
}
