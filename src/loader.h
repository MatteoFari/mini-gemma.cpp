#pragma once
#include <memory>
#include <string>
#include <sys/types.h>
#include "ggml.h"
#include "gguf.h"
#include "ggml-backend.h"

struct Mapping {
    void *data = nullptr;
    size_t size = 0;
    dev_t device;
    ino_t inode;
    explicit Mapping(const std::string &path);
    ~Mapping();
    Mapping(const Mapping &) = delete;
    Mapping &operator=(const Mapping &) = delete;
};

class ModelLoader {
    Mapping mapping_;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> tensors_{nullptr, ggml_free};
    std::unique_ptr<gguf_context, decltype(&gguf_free)> metadata_{nullptr, gguf_free};
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer_{
        nullptr, ggml_backend_buffer_free};

  public:
    explicit ModelLoader(const std::string &path);
    gguf_context *metadata() const { return metadata_.get(); }
    int64_t key(const std::string &name, gguf_type type) const;
    uint32_t integer(const std::string &name) const;
    float number(const std::string &name) const;
    ggml_tensor *tensor(const std::string &name) const;
    size_t bytes() const { return mapping_.size; }
    bool is_model_file(int descriptor) const;
};
