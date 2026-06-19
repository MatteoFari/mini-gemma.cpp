#pragma once
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <vector>

// Builds a graph for one token and runs it with ggml.
class Operations {
    ggml_backend_t backend_ = nullptr;
    ggml_gallocr_t allocator_ = nullptr;
    std::vector<char> arena_;
    std::vector<ggml_backend_buffer_t> external_;
    std::vector<std::pair<ggml_tensor *, std::vector<float>>> masks_;

  public:
    ggml_context *ctx = nullptr;
    ggml_cgraph *graph = nullptr;
    explicit Operations(int threads);
    ~Operations();
    Operations(const Operations &) = delete;
    Operations &operator=(const Operations &) = delete;
    void begin();
    ggml_tensor *external(float *data, int64_t rows, int64_t columns);
    ggml_tensor *mask(int used, int padded);
    ggml_tensor *norm(ggml_tensor *x, ggml_tensor *weight, float epsilon);
    ggml_tensor *multiply(ggml_tensor *weights, ggml_tensor *x);
    void run(ggml_tensor *output, ggml_tensor *token, int token_id, ggml_tensor *pos, int position);
    size_t scratch_bytes() const;
};
