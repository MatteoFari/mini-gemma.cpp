#include "model.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <tuple>

KVCache::KVCache(int w, int c) : width(w), capacity(c), storage((c + 255) / 256 * 256) {
    // Align each allocation for the CPU backend.
    size_t bytes = (size_t(w) * storage * sizeof(float) + 63) / 64 * 64;
    keys.reset(static_cast<float *>(std::aligned_alloc(64, bytes)));
    values.reset(static_cast<float *>(std::aligned_alloc(64, bytes)));
    if (!keys || !values) throw std::bad_alloc();
    std::fill_n(keys.get(), size_t(w) * storage, 0.0f);
    std::fill_n(values.get(), size_t(w) * storage, 0.0f);
}
Model::Model(const ModelLoader &loader, int context, int threads) : ops_(threads), context_(context) {
    if (context < 32 || context > 8192)
        throw std::runtime_error("Context must be between 32 and 8192 tokens");
    auto integer = [&](const char *key) { return int(loader.integer(std::string("gemma4.") + key)); };
    auto number = [&](const char *key) {
        float value = loader.number(std::string("gemma4.") + key);
        if (!std::isfinite(value) || value <= 0) throw std::runtime_error("Invalid model constant");
        return value;
    };
    // Check tensor shapes and types for the pinned E2B model.
    auto weight = [&](const std::string &name, int rows, int columns = 1, ggml_type type = GGML_TYPE_F32) {
        auto *t = loader.tensor(name + ".weight");
        if (t->ne[0] != rows || t->ne[1] != columns || t->ne[2] != 1 || t->ne[3] != 1 || t->type != type)
            throw std::runtime_error("Unsupported tensor layout: " + name);
        return t;
    };
    int count = integer("block_count"), shared = integer("attention.shared_kv_layers");
    if (count != 35 || shared != 20 || integer("embedding_length") != 1536 ||
        integer("attention.head_count_kv") != 1)
        throw std::runtime_error("This demo supports the E2B text model only");
    window_ = integer("attention.sliding_window");
    if (window_ != 512) throw std::runtime_error("Unsupported sliding window");
    epsilon_ = number("attention.layer_norm_rms_epsilon");
    softcap_ = number("final_logit_softcapping");
    base_ = number("rope.freq_base");
    sliding_base_ = number("rope.freq_base_swa");
    per_layer_ = 256;
    embedding_ = weight("token_embd", 1536, 262144, GGML_TYPE_Q6_K);
    layer_embedding_ = weight("per_layer_token_embd", 8960, 262144, GGML_TYPE_Q6_K);
    layer_projection_ = weight("per_layer_model_proj", 1536, 8960, GGML_TYPE_F16);
    layer_norm_ = weight("per_layer_proj_norm", per_layer_);
    output_norm_ = weight("output_norm", 1536);
    rope_factors_ = weight("rope_freqs", 256);
    int last_sliding = -1, last_full = -1;
    for (int i = 0; i < count; ++i) {
        auto t = [&](const char *name, int rows = 1536, int columns = 1, ggml_type type = GGML_TYPE_F32) {
            return weight("blk." + std::to_string(i) + "." + name, rows, columns, type);
        };
        auto matrix = [&](const char *name, int rows, int columns) {
            return t(name, rows, columns, GGML_TYPE_Q4_0);
        };
        LayerWeights l{};
        // Every fifth layer uses full attention.
        l.sliding = i % 5 != 4;
        l.shared = i >= count - shared;
        l.head_dim = l.sliding ? 256 : 512;
        l.heads = 8;
        l.attn_norm = t("attn_norm");
        l.q = matrix("attn_q", 1536, l.head_dim * l.heads);
        l.q_norm = t("attn_q_norm", l.head_dim);
        // Shared layers reuse the last cache of the same attention type.
        if (!l.shared) {
            l.k = matrix("attn_k", 1536, l.head_dim);
            l.k_norm = t("attn_k_norm", l.head_dim);
            l.v = matrix("attn_v", 1536, l.head_dim);
            l.cache = int(caches_.size());
            (l.sliding ? last_sliding : last_full) = l.cache;
            caches_.emplace_back(l.head_dim, l.sliding ? std::min(window_, context) : context);
        } else
            l.cache = l.sliding ? last_sliding : last_full;
        l.out = matrix("attn_output", l.head_dim * l.heads, 1536);
        l.post_attn = t("post_attention_norm");
        l.ffn_norm = t("ffn_norm");
        int hidden = l.shared ? 12288 : 6144;
        l.gate = matrix("ffn_gate", 1536, hidden);
        l.up = matrix("ffn_up", 1536, hidden);
        l.down = matrix("ffn_down", hidden, 1536);
        l.post_ffn = t("post_ffw_norm");
        l.input_gate = matrix("inp_gate", 1536, per_layer_);
        l.projection = matrix("proj", per_layer_, 1536);
        l.post_norm = t("post_norm");
        l.scale = *static_cast<float *>(t("layer_output_scale", 1)->data);
        if (!std::isfinite(l.scale)) throw std::runtime_error("Invalid layer scale");
        layers_.push_back(l);
    }
}
std::span<const float> Model::forward(int token_id, bool output) {
    if (token_id < 0 || token_id >= embedding_->ne[1]) throw std::runtime_error("Token outside vocabulary");
    if (position_ >= context_) throw std::runtime_error("Context full; use /reset");
    ops_.begin();
    auto *c = ops_.ctx;
    auto *token = ggml_new_tensor_1d(c, GGML_TYPE_I32, 1);
    auto *pos = ggml_new_tensor_1d(c, GGML_TYPE_I32, 1);
    ggml_set_input(token);
    ggml_set_input(pos);
    auto mm = [&](ggml_tensor *w, ggml_tensor *x) { return ops_.multiply(w, x); };
    auto norm = [&](ggml_tensor *x, ggml_tensor *w) { return ops_.norm(x, w, epsilon_); };
    // Prepare the token embedding and per-layer inputs.
    auto *x = ggml_scale(c, ggml_get_rows(c, embedding_, token), std::sqrt(float(embedding_->ne[0])));
    auto *projected = ggml_scale(c, mm(layer_projection_, x), 1.0f / std::sqrt(float(embedding_->ne[0])));
    auto *per_token = ggml_scale(c, ggml_get_rows(c, layer_embedding_, token), std::sqrt(float(per_layer_)));
    std::vector<std::tuple<ggml_tensor *, ggml_tensor *, ggml_tensor *>> cache_views(caches_.size());
    for (size_t i = 0; i < layers_.size(); ++i) {
        const auto &l = layers_[i];
        auto rope = [&](ggml_tensor *value) {
            return ggml_rope_ext(c, value, pos, l.sliding ? nullptr : rope_factors_, l.head_dim,
                                 GGML_ROPE_TYPE_NEOX, 0, l.sliding ? sliding_base_ : base_, 1, 0, 1, 0, 0);
        };
        // Compute K/V once and reuse them in shared layers.
        auto *input = norm(x, l.attn_norm);
        auto *q = rope(norm(ggml_reshape_2d(c, mm(l.q, input), l.head_dim, l.heads), l.q_norm));
        if (!l.shared) {
            auto &cache = caches_[l.cache];
            int used = std::min(position_ + 1, cache.capacity);
            int padded = (used + 255) / 256 * 256;
            size_t offset = size_t(KVCache::slot(position_, cache.capacity)) * cache.width * sizeof(float);
            auto *keys = ops_.external(cache.keys.get(), cache.width, padded);
            auto *values = ops_.external(cache.values.get(), cache.width, padded);
            auto *k = rope(norm(mm(l.k, input), l.k_norm));
            auto *v = norm(mm(l.v, input), nullptr);
            // Schedule cache writes before attention reads.
            ggml_build_forward_expand(ops_.graph, ggml_cpy(c, k, ggml_view_1d(c, keys, cache.width, offset)));
            ggml_build_forward_expand(ops_.graph,
                                      ggml_cpy(c, v, ggml_view_1d(c, values, cache.width, offset)));
            cache_views[l.cache] = {keys, values, ops_.mask(used, padded)};
        }
        auto [keys, values, mask] = cache_views[l.cache];
        q = ggml_reshape_3d(c, q, l.head_dim, 1, l.heads);
        // Masked padding keeps SIMD softmax stable as context grows.
        auto *scores = mm(keys, q);
        ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
        scores = ggml_soft_max_ext(c, scores, mask, 1.0f, 0.0f);
        auto *attended = mm(ggml_cont(c, ggml_transpose(c, values)), scores);
        auto *flat = ggml_reshape_1d(c, attended, int64_t(l.head_dim) * l.heads);
        x = ggml_add(c, x, norm(mm(l.out, flat), l.post_attn));
        // Apply GELU gating, then add the residual.
        input = norm(x, l.ffn_norm);
        auto *gated = ggml_mul(c, ggml_gelu(c, mm(l.gate, input)), mm(l.up, input));
        x = ggml_add(c, x, norm(mm(l.down, gated), l.post_ffn));
        // Add Gemma's token-dependent input to this layer.
        size_t offset = i * per_layer_ * sizeof(float);
        auto *extra =
            ggml_scale(c,
                       ggml_add(c, norm(ggml_view_1d(c, projected, per_layer_, offset), layer_norm_),
                                ggml_view_1d(c, per_token, per_layer_, offset)),
                       1.0f / std::sqrt(2.0f));
        auto *gate = ggml_mul(c, ggml_gelu(c, mm(l.input_gate, x)), extra);
        x = ggml_scale(c, ggml_add(c, x, norm(mm(l.projection, gate), l.post_norm)), l.scale);
    }
    if (output) {
        // Reuse embedding weights for logits, then apply the soft cap.
        x = mm(embedding_, norm(x, output_norm_));
        x = ggml_scale(c, ggml_tanh(c, ggml_scale(c, x, 1.0f / softcap_)), softcap_);
    }
    ops_.run(x, token, token_id, pos, position_);
    ++position_;
    return {static_cast<const float *>(x->data), output ? size_t(embedding_->ne[1]) : 0};
}
size_t Model::cache_bytes() const {
    size_t bytes = 0;
    for (const auto &c : caches_)
        bytes += size_t(c.width) * c.storage * 2 * sizeof(float);
    return bytes;
}
size_t Model::snapshot_bytes() const {
    size_t bytes = 0;
    for (const auto &c : caches_)
        bytes += size_t(c.width) * std::min(position_, c.capacity) * 2 * sizeof(float);
    return bytes;
}
Model::Snapshot Model::snapshot() const {
    Snapshot saved(this, position_, snapshot_bytes() / sizeof(float));
    auto output = saved.data_.begin();
    for (const auto &c : caches_) {
        size_t active = size_t(c.width) * std::min(position_, c.capacity);
        // Keep ring order unchanged to preserve floating-point results.
        output = std::copy_n(c.keys.get(), active, output);
        output = std::copy_n(c.values.get(), active, output);
    }
    return saved;
}
void Model::restore(const Snapshot &saved) {
    if (saved.owner_ != this || saved.position_ < 0 || saved.position_ > context_)
        throw std::runtime_error("Snapshot belongs to a different model");
    size_t expected = 0;
    for (const auto &c : caches_)
        expected += size_t(c.width) * std::min(saved.position_, c.capacity) * 2;
    if (saved.data_.size() != expected) throw std::runtime_error("Invalid snapshot storage");
    auto input = saved.data_.begin();
    for (auto &c : caches_) {
        size_t active = size_t(c.width) * std::min(saved.position_, c.capacity);
        for (float *buffer : {c.keys.get(), c.values.get()}) {
            std::copy_n(input, active, buffer);
            input += active;
            // Clear padding so attention cannot read stale values.
            std::fill(buffer + active, buffer + size_t(c.width) * c.storage, 0.0f);
        }
    }
    position_ = saved.position_;
}
