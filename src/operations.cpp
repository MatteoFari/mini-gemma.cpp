#include "operations.h"
#include "ggml-cpu.h"
#include <stdexcept>
#include <limits>
#include <algorithm>

Operations::Operations(int threads) : arena_(8 * 1024 * 1024) {
    if (threads < 1 || threads > 256) throw std::runtime_error("Invalid thread count");
    backend_ = ggml_backend_cpu_init();
    if (!backend_) throw std::runtime_error("CPU backend initialization failed");
    ggml_backend_cpu_set_n_threads(backend_, threads);
    allocator_ = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
}
Operations::~Operations() {
    if (ctx) ggml_free(ctx);
    for (auto *buffer : external_)
        ggml_backend_buffer_free(buffer);
    ggml_gallocr_free(allocator_);
    ggml_backend_free(backend_);
}
void Operations::begin() {
    // Rebuild graph metadata while keeping the scratch allocator.
    if (ctx) ggml_free(ctx);
    for (auto *buffer : external_)
        ggml_backend_buffer_free(buffer);
    external_.clear();
    masks_.clear();
    ctx = ggml_init({arena_.size(), arena_.data(), true});
    if (!ctx) throw std::runtime_error("Graph context allocation failed");
    graph = ggml_new_graph_custom(ctx, 8192, false);
}
ggml_tensor *Operations::external(float *data, int64_t rows, int64_t columns) {
    // Wrap existing CPU memory without copying it.
    auto *t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, rows, columns);
    auto *buffer = ggml_backend_cpu_buffer_from_ptr(data, ggml_nbytes(t));
    if (!buffer) throw std::runtime_error("Cache buffer allocation failed");
    external_.push_back(buffer);
    t->data = data;
    t->buffer = buffer;
    return t;
}
ggml_tensor *Operations::norm(ggml_tensor *x, ggml_tensor *weight, float epsilon) {
    auto *result = ggml_rms_norm(ctx, x, epsilon);
    return weight ? ggml_mul(ctx, result, weight) : result;
}
ggml_tensor *Operations::mask(int used, int padded) {
    auto *t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, padded);
    ggml_set_input(t);
    // Hide unused cache slots with negative infinity.
    std::vector<float> values(padded, -std::numeric_limits<float>::infinity());
    std::fill_n(values.begin(), used, 0.0f);
    masks_.emplace_back(t, std::move(values));
    return t;
}
ggml_tensor *Operations::multiply(ggml_tensor *weights, ggml_tensor *x) {
    return ggml_mul_mat(ctx, weights, x);
}
void Operations::run(ggml_tensor *output, ggml_tensor *token, int token_id, ggml_tensor *pos, int position) {
    ggml_set_output(output);
    ggml_build_forward_expand(graph, output);
    // Allocate graph buffers before setting the token and position.
    if (!ggml_gallocr_alloc_graph(allocator_, graph)) throw std::runtime_error("Graph allocation failed");
    ggml_backend_tensor_set(token, &token_id, 0, sizeof(int32_t));
    ggml_backend_tensor_set(pos, &position, 0, sizeof(int32_t));
    for (auto &[tensor, values] : masks_)
        ggml_backend_tensor_set(tensor, values.data(), 0, values.size() * sizeof(float));
    if (ggml_backend_graph_compute(backend_, graph) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("CPU graph execution failed");
}
size_t Operations::scratch_bytes() const {
    return ggml_gallocr_get_buffer_size(allocator_, 0);
}
