#pragma once
#include "model.h"
#include <list>

// Reuses model state when prompt tokens match a saved prefix.
class PrefixCache {
    struct Entry {
        std::vector<int> tokens;
        Model::Snapshot state;
        size_t bytes() const { return sizeof(Entry) + tokens.capacity() * sizeof(int) + state.bytes(); }
    };
    Model &model_;
    size_t budget_, bytes_ = 0, evictions_ = 0;
    // Most recently used checkpoints stay at the front.
    std::list<Entry> entries_;
    void evict();

  public:
    PrefixCache(Model &model, size_t budget) : model_(model), budget_(budget) {}
    PrefixCache(const PrefixCache &) = delete;
    PrefixCache &operator=(const PrefixCache &) = delete;
    size_t restore(std::span<const int> tokens);
    void remember(std::span<const int> prefix);
    bool enabled() const { return budget_ != 0; }
    size_t bytes() const { return bytes_; }
    size_t budget() const { return budget_; }
    size_t evictions() const { return evictions_; }
};
