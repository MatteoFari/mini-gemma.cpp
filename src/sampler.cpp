#include "sampler.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

int sample(std::span<const float> logits) {
    if (logits.empty()) throw std::runtime_error("Empty model logits");
    for (float value : logits)
        if (!std::isfinite(value)) throw std::runtime_error("Non-finite model logits");
    return int(std::max_element(logits.begin(), logits.end()) - logits.begin());
}
