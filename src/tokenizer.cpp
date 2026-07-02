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
std::vector<int> Tokenizer::encode(std::string_view text, bool initial) const {
    if (text.size() > size_t(std::numeric_limits<int>::max() / 8))
        throw std::runtime_error("Prompt too large");
    struct Piece {
        int prev, next, start, length;
        bool alive = true;
    };
    // Lower merge ranks win. Ties go to the leftmost pair.
    using Merge = std::tuple<int, int, int, int, int>;
    std::priority_queue<Merge, std::vector<Merge>, std::greater<Merge>> queue;
    std::vector<Piece> pieces;
    std::string normalized;
    auto propose = [&](int left, int right) {
        if (left < 0 || right < 0) return;
        const auto &a = pieces[left];
        const auto &b = pieces[right];
        std::string pair = normalized.substr(a.start, a.length) + " " + normalized.substr(b.start, b.length);
        auto it = ranks_.find(pair);
        if (it != ranks_.end()) queue.emplace(it->second, left, right, a.length, b.length);
    };
    auto append = [&](std::string_view value) {
        int i = int(pieces.size());
        pieces.push_back({i - 1, -1, int(normalized.size()), int(value.size())});
        normalized.append(value);
        if (i) {
            pieces[i - 1].next = i;
            propose(i - 1, i);
        }
    };
    if (space_prefix_ && !text.empty() && text.front() != ' ') append(space);
    // Split text into control tokens, characters, or byte fallbacks.
    for (size_t i = 0; i < text.size();) {
        bool special = false;
        if (text[i] == '<')
            for (auto control : controls_) {
                if (text.substr(i).starts_with(control)) {
                    append(control);
                    i += control.size();
                    special = true;
                    break;
                }
            }
        if (special) continue;
        auto byte = static_cast<unsigned char>(text[i]);
        size_t count = byte < 0x80 ? 1 : byte < 0xE0 ? 2 : byte < 0xF0 ? 3 : 4;
        count = std::min(count, text.size() - i);
        auto character = text.substr(i, count);
        if (character == " ")
            append(space);
        else if (ids_.contains(character))
            append(character);
        else
            for (unsigned char part : character) {
                char fallback[7];
                std::snprintf(fallback, sizeof(fallback), "<0x%02X>", part);
                append(fallback);
            }
        i += count;
    }
    // Merge the best-ranked adjacent pieces until no candidates remain.
    while (!queue.empty()) {
        auto [rank, left, right, left_size, right_size] = queue.top();
        queue.pop();
        auto &a = pieces[left];
        auto &b = pieces[right];
        // Ignore pairs changed by an earlier merge.
        if (!a.alive || !b.alive || a.next != right || a.length != left_size || b.length != right_size)
            continue;
        a.length += b.length;
        a.next = b.next;
        b.alive = false;
        if (b.next >= 0) pieces[b.next].prev = left;
        propose(a.prev, left);
        propose(left, a.next);
    }
    std::vector<int> result;
    if (initial && add_bos_) result.push_back(bos_);
    for (int i = pieces.empty() ? -1 : 0; i >= 0; i = pieces[i].next) {
        auto value = std::string_view(normalized).substr(pieces[i].start, pieces[i].length);
        auto it = ids_.find(value);
        result.push_back(it == ids_.end() ? unknown_ : it->second);
    }
    return result;
}
