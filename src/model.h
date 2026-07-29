#pragma once
#include "loader.h"
#include "operations.h"
#include <span>
#include <cstdlib>
#include <vector>

struct LayerWeights {
    bool sliding, shared;
    int cache, head_dim, heads;
    float scale;
    ggml_tensor *attn_norm, *q, *q_norm, *k = nullptr, *k_norm = nullptr, *v = nullptr, *out, *post_attn;
    ggml_tensor *ffn_norm, *gate, *up, *down, *post_ffn, *input_gate, *projection, *post_norm;
};
struct KVCache {
    int width, capacity, storage;
    std::unique_ptr<float[], decltype(&std::free)> keys{nullptr, std::free}, values{nullptr, std::free};
    KVCache(int width, int capacity);
    static int slot(int position, int capacity) { return position % capacity; }
};
class Model {
    Operations ops_;
    std::vector<LayerWeights> layers_;
    std::vector<KVCache> caches_;
    int position_ = 0, context_, window_, per_layer_;
    float epsilon_, softcap_, base_, sliding_base_;
    ggml_tensor *embedding_, *layer_embedding_, *layer_projection_, *layer_norm_, *output_norm_,
        *rope_factors_;

  public:
    class Snapshot {
        friend class Model;
        const Model *owner_;
        int position_;
        std::vector<float> data_;
        Snapshot(const Model *owner, int position, size_t elements)
            : owner_(owner), position_(position), data_(elements) {}

      public:
        size_t bytes() const { return data_.capacity() * sizeof(float); }
    };
    Model(const ModelLoader &loader, int context, int threads);
    // Returned logits are valid until the next forward pass.
    std::span<const float> forward(int token, bool output = true);
    void reset() { position_ = 0; }
    int position() const { return position_; }
    int capacity() const { return context_; }
    size_t cache_bytes() const;
    size_t snapshot_bytes() const;
    Snapshot snapshot() const;
    void restore(const Snapshot &snapshot);
    size_t scratch_bytes() const { return ops_.scratch_bytes(); }
};
