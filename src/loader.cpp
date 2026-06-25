#include "loader.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdexcept>

Mapping::Mapping(const std::string &path) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("Cannot open model: " + path);
    struct stat info{};
    if (fstat(fd, &info) != 0 || info.st_size < 24) {
        close(fd);
        throw std::runtime_error("Empty or unreadable model");
    }
    size = static_cast<size_t>(info.st_size);
    device = info.st_dev;
    inode = info.st_ino;
    // Map the weights read-only without copying them.
    data = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (data == MAP_FAILED) throw std::runtime_error("Could not memory-map model");
}
Mapping::~Mapping() {
    if (data && data != MAP_FAILED) munmap(data, size);
}

ModelLoader::ModelLoader(const std::string &path) : mapping_(path) {
    ggml_context *ctx = nullptr;
    // Read tensor metadata without allocating weight data.
    metadata_.reset(gguf_init_from_file(path.c_str(), {true, &ctx}));
    tensors_.reset(ctx);
    if (!metadata_ || !tensors_) throw std::runtime_error("Invalid GGUF model");
    auto arch = key("general.architecture", GGUF_TYPE_STRING);
    if (std::string(gguf_get_val_str(metadata(), arch)) != "gemma4")
        throw std::runtime_error("Only the Gemma4 E2B text architecture is supported");
    if (gguf_get_alignment(metadata()) < 32) throw std::runtime_error("Unsupported tensor alignment");
    buffer_.reset(ggml_backend_cpu_buffer_from_ptr(mapping_.data, mapping_.size));
    if (!buffer_) throw std::runtime_error("Cannot wrap model memory");
    // Point each tensor into the mapped weight data.
    size_t base = gguf_get_data_offset(metadata());
    for (int64_t i = 0; i < gguf_get_n_tensors(metadata()); ++i) {
        auto *t = tensor(gguf_get_tensor_name(metadata(), i));
        size_t offset = gguf_get_tensor_offset(metadata(), i);
        if (base > mapping_.size || offset > mapping_.size - base ||
            ggml_nbytes(t) > mapping_.size - base - offset)
            throw std::runtime_error("Truncated tensor data");
        t->data = static_cast<char *>(mapping_.data) + base + offset;
        t->buffer = buffer_.get();
    }
}
int64_t ModelLoader::key(const std::string &name, gguf_type type) const {
    auto id = gguf_find_key(metadata(), name.c_str());
    if (id < 0 || gguf_get_kv_type(metadata(), id) != type)
        throw std::runtime_error("Missing or invalid metadata: " + name);
    return id;
}
uint32_t ModelLoader::integer(const std::string &name) const {
    return gguf_get_val_u32(metadata(), key(name, GGUF_TYPE_UINT32));
}
float ModelLoader::number(const std::string &name) const {
    return gguf_get_val_f32(metadata(), key(name, GGUF_TYPE_FLOAT32));
}
ggml_tensor *ModelLoader::tensor(const std::string &name) const {
    auto *t = ggml_get_tensor(tensors_.get(), name.c_str());
    if (!t) throw std::runtime_error("Missing tensor: " + name);
    return t;
}
bool ModelLoader::is_model_file(int descriptor) const {
    // Compare file identity so links are caught too.
    struct stat info{};
    if (fstat(descriptor, &info) != 0 || !S_ISREG(info.st_mode))
        throw std::runtime_error("Logits output must be a regular file");
    return info.st_dev == mapping_.device && info.st_ino == mapping_.inode;
}
