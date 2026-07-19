# mini-gemma.cpp

A compact C++20 CPU inference engine for the pinned Gemma 4 E2B GGUF model.
Uses ggml for numerical kernels and implements tokenization, transformer execution,
KV caching, greedy or temperature/top-p sampling, and terminal conversations.

Requires macOS or Linux, CMake 3.24+, a C++20 compiler, and Python 3.11+.

```sh
python3 scripts/download_model.py
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 4
./build/inference_engine models/gemma-4-E2B_q4_0-it.gguf
```

Use `/metrics`, `/reset`, `/help`, and `/exit` during a conversation.
Pass `--prompt "Your question"` for a single response or `--help` for all options.
See [Roadmap.md](Roadmap.md) for implementation progress.
