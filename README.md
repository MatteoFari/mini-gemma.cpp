# mini-gemma.cpp

A compact C++20 CPU inference engine for Gemma 4 E2B. It memory-maps quantized GGUF weights and runs text generation end to end, with BPE tokenization, transformer execution, KV caching, and streamed terminal responses. The model graph and inference loop are implemented in C++. **ggml** provides numerical operations, CPU execution, and GGUF parsing.

```text
User > Remember: my name is Matteo. Reply with OK.
AI > OK
User > What is my name? Answer with just the name.
AI > Matteo
```

## Capabilities

Interactive conversations support greedy or temperature/top-p sampling, full and sliding-window attention, and shared KV caches. Optional prefix checkpoints reuse prompt computation across conversation resets, with a configurable memory budget and LRU eviction.

In a local benchmark with six different questions about a shared document, prefix caching reduced median prompt-processing time from **6.95 s to 0.39 s (~18× faster)** with identical generated responses.

The engine targets one pinned Gemma 4 E2B quantized model configuration and runs text-only inference on the CPU. See [Roadmap.md](Roadmap.md) for implementation milestones and planned improvements.

## Source Layout

```text
mini-gemma-cpp/
├── CMakeLists.txt          # Build configuration and ggml dependency
├── Roadmap.md              # Implementation milestones and next steps
├── src/                    # Headers and implementations together
│   ├── engine.{h,cpp}       # Conversation state and generation
│   ├── loader.{h,cpp}       # Memory-mapped GGUF weights
│   ├── model.{h,cpp}        # Transformer layers and KV state
│   ├── operations.{h,cpp}   # ggml graph execution
│   ├── prefix_cache.{h,cpp} # Prefix checkpoints and LRU eviction
│   ├── profiler.{h,cpp}     # Timing and memory metrics
│   ├── sampler.{h,cpp}      # Greedy and temperature/top-p sampling
│   ├── tokenizer.{h,cpp}    # BPE encoding and decoding
│   └── main.cpp            # CLI entry point
├── tests/                  # Functional checks and reference data
└── scripts/
    ├── download_model.py   # Download and verify model weights
    └── benchmark_prefix.py # Measure prefix reuse
```

## Quick Start

Requires macOS or Linux, a C++20 compiler, CMake 3.24+, and Python 3.11+. Run from the project root:

```sh
python3 scripts/download_model.py
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 4
./build/inference_engine models/gemma-4-E2B_q4_0-it.gguf
```

The download script fetches the supported GGUF (about 3.35 GB) and verifies its checksum. Tokenizer data is included in the file. CMake downloads a pinned llama.cpp archive and builds only its ggml subtree.

Add `--prefix-cache-mib 128` to retain up to 128 MiB of prefix checkpoints, or `--prompt "Your question"` for a single response. Run `./build/inference_engine --help` to see all options.

## Chat Commands

| Command | Action |
| --- | --- |
| `/metrics` | Show latency, throughput, reused tokens, and allocated cache memory |
| `/reset` | Clear the conversation and reset sampling while retaining prefix checkpoints |
| `/help` | List available options and commands |
| `/exit` | Quit |
