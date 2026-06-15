# mini-gemma.cpp Roadmap

Build a working inference path one component at a time: CPU operations, model loading, tokenization, transformer layers, and generation. Keep each component in its own module so problems can be isolated before connecting the full chat loop. Use ggml for numerical kernels and target the pinned Gemma 4 E2B GGUF.

## 1. CPU Operations (`operations.cpp`)

Set up the numerical backend that the transformer will use.

- [ ] **CPU backend:** Initialize ggml and configure the number of worker threads.
- [ ] **Tensor operations:** Wrap matrix multiplication, RMS normalization, and KV buffer access with attention masking.
- [ ] **Graph execution:** Build and execute a graph for each token while reusing scratch allocations.

## 2. Model Loading (`loader.cpp`, `model.cpp`)

Read the model configuration and access quantized weights without copying the entire file into new buffers.

- [ ] **Weight access:** Read GGUF metadata, map weights read-only, and bind tensors to their file offsets with automatic resource cleanup.
- [ ] **Validation:** Reject truncated data and check the architecture, required tensor shapes, and numerical formats.

## 3. Text Encoding (`tokenizer.cpp`)

Translate prompts into token IDs and generated IDs back into readable text.

- [ ] **Vocabulary:** Load tokens, merge ranks, and special-token settings from the GGUF.
- [ ] **BPE encoding:** Merge pieces by rank with deterministic ties, UTF-8 handling, byte fallback, and control-token support.
- [ ] **Decoding:** Restore text from token IDs and recognize end-of-sequence and turn markers.

## 4. Transformer Layers (`model.cpp`)

Assemble Gemma's attention and feed-forward blocks from ggml operations.

- [ ] **Embeddings:** Look up token embeddings and prepare the additional per-layer inputs.
- [ ] **Attention:** Build query/key/value projections, apply normalization and RoPE, and compute full or sliding-window attention with masked softmax.
- [ ] **Feed-forward blocks:** Add GELU gating, per-layer input projections, normalization, and residual connections.

## 5. Forward Pass and KV State (`model.cpp`)

Connect the layers and retain the state needed to process the next token.

- [ ] **KV storage:** Allocate aligned F32 buffers for full and sliding attention, with ring indexing and cache sharing between the appropriate layers.
- [ ] **Forward pass:** Run all 35 layers, update cache positions, and produce soft-capped logits, skipping the vocabulary projection for intermediate prompt tokens.
- [ ] **Batched prefill:** Process several prompt tokens per graph to reduce execution overhead.

## 6. Sampling and Metrics (`sampler.cpp`, `profiler.cpp`)

Turn model scores into token choices and measure the cost of inference.

- [ ] **Token selection:** Support greedy and temperature/top-p sampling with seeded randomness.
- [ ] **Profiling:** Report initialization, prefill, first-token latency, decode throughput, token counts, and allocated KV/scratch memory.
- [ ] **Top-k sampling:** Add an optional candidate limit before top-p selection.

## 7. Conversation Management (`engine.cpp`)

Coordinate the loader, tokenizer, model, sampling, and profiling in the engine.

- [ ] **Generation loop:** Format Gemma's chat turns, process the prompt, and stream tokens until a stop marker or output limit.
- [ ] **Conversation state:** Preserve previous turns and close both natural and length-limited replies.
- [ ] **Context management:** Reserve space for the prompt, reply, and closing tokens before changing state. Support clearing the conversation and reseeding sampling.

## 8. Terminal Interface (`main.cpp`)

Expose the engine through a small interactive CLI.

- [ ] **Interactive use:** Read prompts, stream responses, and handle `/metrics`, `/reset`, `/help`, and `/exit`.
- [ ] **CLI options:** Expose model and generation settings, one-shot prompts, token inspection, and raw-logit export.
