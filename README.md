# qwen-rs

Inference engine for **Qwen2.5-0.5B-Instruct**, written from scratch in Rust. It uses no ML frameworks (no candle, burn, tch or ort). Weights are loaded straight from safetensors, and every op in the forward pass is implemented by hand.

> 🚧 **Work in progress.** v1 (correct generation, KV cache, multithreading, benchmarks) is targeted for January 2027.

## Goals

- Run autoregressive text generation for Qwen2.5-0.5B-Instruct entirely in Rust
- **Correctness first:** logits must match Hugging Face `transformers` (PyTorch, fp32) within tolerance on fixed prompts, checked layer by layer
- Then optimize, and benchmark tokens/sec, time-to-first-token and peak memory against PyTorch (eager, CPU) and llama.cpp on an Apple M3

## Model

| | |
|---|---|
| Model | Qwen2.5-0.5B-Instruct (~494M params, bf16 safetensors) |
| Layers | 24, hidden size 896, MLP size 4864 |
| Attention | GQA: 14 query heads, 2 KV heads, head dim 64, QKV biases |
| Position | RoPE, θ = 1,000,000 |
| Vocab | 151,936; input embeddings and output layer share weights |

## Roadmap

- [x] **M0:** Toolchain, pinned weights, HF reference fixtures, safetensors inspector
- [ ] **M1:** Config + safetensors loader (bf16 → f32), tensor type, tokenizer
- [ ] **M2:** Forward pass: RMSNorm, RoPE, GQA attention, SwiGLU; logits match HF
- [ ] **M3:** Sampling (greedy / temperature / top-p) + streaming CLI
- [ ] **M4:** KV cache
- [ ] **M5:** Multithreaded matmul (rayon)
- [ ] **M6:** Benchmarks vs PyTorch and llama.cpp
- [ ] Post-v1: NEON SIMD → int8 weight quantization → int4

## Setup

Requires Rust 1.99.1 (pinned in `rust-toolchain.toml`) and Python 3 for the reference scripts.

```bash
# Weights (pinned revision)
python -m venv .venv && source .venv/bin/activate
pip install -r scripts/requirements.txt
hf download Qwen/Qwen2.5-0.5B-Instruct --revision $(cat models/REVISION) \
  --local-dir models/qwen2.5-0.5b-instruct

# Reference activations + logits from HF transformers
python scripts/dump_reference.py --revision $(cat models/REVISION)

# Inspect the weight file
cargo run --release --bin inspect -- models/qwen2.5-0.5b-instruct/model.safetensors
```

## Layout

```
src/            engine (loader, tensor, ops, model, sampler, CLI)
src/bin/        dev tools (safetensors inspector)
tests/          reference-comparison tests against HF fixtures
scripts/        Python reference + benchmark scripts
fixtures/       manifest.json (committed); activation dumps (generated, gitignored)
```
