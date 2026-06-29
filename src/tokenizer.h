#pragma once
#include "loader.h"
#include <string_view>
#include <unordered_map>
#include <vector>

class Tokenizer {
    // Vocabulary strings are borrowed from the loader.
    std::vector<std::string_view> tokens_, controls_;
    std::unordered_map<std::string_view, int> ids_, ranks_;
    int bos_, eos_, unknown_;
    bool add_bos_, space_prefix_;

  public:
    explicit Tokenizer(const ModelLoader &loader);
    std::vector<int> encode(std::string_view text, bool initial = true) const;
    std::string decode(int token) const;
    bool stop(int token) const;
};
