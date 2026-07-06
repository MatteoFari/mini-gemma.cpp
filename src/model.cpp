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
