#include "prefix_cache.h"
#include <algorithm>
#include <stdexcept>

void PrefixCache::evict() {
    bytes_ -= entries_.back().bytes();
    entries_.pop_back();
    ++evictions_;
}
size_t PrefixCache::restore(std::span<const int> tokens) {
    if (model_.position() != 0) throw std::runtime_error("Prefix restore requires an empty conversation");
    auto best = entries_.end();
    // Find the longest exact token prefix.
    for (auto entry = entries_.begin(); entry != entries_.end(); ++entry) {
        size_t n = entry->tokens.size();
        // Leave one prompt token to recompute logits.
        if (n >= tokens.size() || (best != entries_.end() && n <= best->tokens.size())) continue;
        if (std::equal(entry->tokens.begin(), entry->tokens.end(), tokens.begin())) best = entry;
    }
    if (best == entries_.end()) return 0;
    model_.restore(best->state);
    entries_.splice(entries_.begin(), entries_, best);
    return entries_.front().tokens.size();
}
void PrefixCache::remember(std::span<const int> prefix) {
    if (!enabled() || prefix.empty()) return;
    if (prefix.size() != size_t(model_.position()))
        throw std::runtime_error("Checkpoint length does not match model position");
    // Refresh existing checkpoints without copying their state.
    for (auto entry = entries_.begin(); entry != entries_.end(); ++entry) {
        if (std::equal(prefix.begin(), prefix.end(), entry->tokens.begin(), entry->tokens.end())) {
            entries_.splice(entries_.begin(), entries_, entry);
            return;
        }
    }
    // Reject oversized checkpoints before copying any KV data.
    size_t estimate = sizeof(Entry) + prefix.size_bytes() + model_.snapshot_bytes();
    if (estimate > budget_) return;
    while (bytes_ > budget_ - estimate)
        evict();
    Entry entry{{prefix.begin(), prefix.end()}, model_.snapshot()};
    size_t actual = entry.bytes();
    if (actual > budget_) return;
    while (bytes_ > budget_ - actual)
        evict();
    entries_.push_front(std::move(entry));
    bytes_ += actual;
}
