#include "tokenizer.h"
#include <algorithm>
#include <charconv>
#include <cstdio>
#include <limits>
#include <queue>
#include <stdexcept>
#include <tuple>

namespace {
// U+2581 is the vocabulary's space marker.
constexpr std::string_view space = "\xE2\x96\x81";
}
Tokenizer::Tokenizer(const ModelLoader &loader) {
    auto *meta = loader.metadata();
    auto integer = [&](const char *name) {
        auto id = loader.integer(std::string("tokenizer.ggml.") + name);
        if (id > uint32_t(std::numeric_limits<int>::max()))
            throw std::runtime_error("Invalid special token ID");
        return int(id);
    };
    bos_ = integer("bos_token_id");
    eos_ = integer("eos_token_id");
    unknown_ = integer("unknown_token_id");
    add_bos_ = gguf_get_val_bool(meta, loader.key("tokenizer.ggml.add_bos_token", GGUF_TYPE_BOOL));
    space_prefix_ = gguf_get_val_bool(meta, loader.key("tokenizer.ggml.add_space_prefix", GGUF_TYPE_BOOL));
    auto vocab = loader.key("tokenizer.ggml.tokens", GGUF_TYPE_ARRAY);
    auto types = loader.key("tokenizer.ggml.token_type", GGUF_TYPE_ARRAY);
    auto merges = loader.key("tokenizer.ggml.merges", GGUF_TYPE_ARRAY);
    if (gguf_get_arr_type(meta, vocab) != GGUF_TYPE_STRING ||
        gguf_get_arr_type(meta, types) != GGUF_TYPE_INT32 ||
        gguf_get_arr_type(meta, merges) != GGUF_TYPE_STRING ||
        gguf_get_arr_n(meta, vocab) != gguf_get_arr_n(meta, types))
        throw std::runtime_error("Invalid tokenizer arrays");
    size_t count = gguf_get_arr_n(meta, vocab);
    if (count == 0 || count > 262144 || size_t(std::max({bos_, eos_, unknown_})) >= count)
        throw std::runtime_error("Invalid vocabulary size or special token ID");
    auto *token_types = static_cast<const int32_t *>(gguf_get_arr_data(meta, types));
    for (size_t i = 0; i < gguf_get_arr_n(meta, vocab); ++i) {
        std::string_view value = gguf_get_arr_str(meta, vocab, i);
        tokens_.push_back(value);
        ids_[value] = int(i);
        if (token_types[i] == 3 || token_types[i] == 4) {
            if (value.empty()) throw std::runtime_error("Empty control token");
            controls_.push_back(value);
        }
    }
    // Match longer control tokens before their prefixes.
    std::sort(controls_.begin(), controls_.end(), [](auto a, auto b) { return a.size() > b.size(); });
    for (size_t i = 0; i < gguf_get_arr_n(meta, merges); ++i)
        ranks_[gguf_get_arr_str(meta, merges, i)] = int(i);
}
