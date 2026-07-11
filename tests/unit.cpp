#include "operations.h"
#include "sampler.h"
#include "model.h"
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>

void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}
int main() try {
    std::array<float, 3> logits{1, 4, 2};
    require(sample(logits) == 1, "Greedy chooses maximum");
    // Exercise actual ggml graph allocation, projection and normalization against known values.
    Operations ops(2);
    alignas(64) std::array<float, 4> embedding{3, 4, 8, 6};
    alignas(64) std::array<float, 4> identity{1, 0, 0, 1};
    ops.begin();
    auto *c = ops.ctx;
    auto *token = ggml_new_tensor_1d(c, GGML_TYPE_I32, 1);
    auto *pos = ggml_new_tensor_1d(c, GGML_TYPE_I32, 1);
    ggml_set_input(token);
    ggml_set_input(pos);
    auto *x = ggml_get_rows(c, ops.external(embedding.data(), 2, 2), token);
    auto *projected = ops.multiply(ops.external(identity.data(), 2, 2), x);
    auto *result =
        ggml_add(c, ops.norm(projected, nullptr, 1e-6f), ggml_scale(c, ggml_cast(c, pos, GGML_TYPE_F32), 0));
    ops.run(result, token, 0, pos, 0);
    auto *values = static_cast<float *>(result->data);
    require(std::abs(values[0] - 3.0f / std::sqrt(12.5f)) < 1e-5f, "RMSNorm first component");
    require(std::abs(values[1] - 4.0f / std::sqrt(12.5f)) < 1e-5f, "RMSNorm second component");
    // A full ring must contain exactly the most recent 512 positions after wrapping.
    std::array<int, 512> ring{};
    for (int position = 0; position < 600; ++position)
        ring[KVCache::slot(position, 512)] = position;
    for (int position = 88; position < 600; ++position)
        require(ring[KVCache::slot(position, 512)] == position, "Sliding cache retained recent position");
    std::cout << "Sampling, ggml operations and cache indexing passed.\n";
    return 0;
} catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
}
