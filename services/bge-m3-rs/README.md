# bge-m3-rs

CPU-only Rust port of `../bge-m3/serve.py`. Same wire contract, so it's a drop-in
for `BGE_M3_URL`:

```
POST /embed   {"texts": ["...", ...]}  ->  {"embeddings": [[...; 1024], ...]}
GET  /health
```

One 1024-dim, L2-normalized, CLS-pooled BGE-M3 dense vector per text, in order.

## Why this exists

`serve.py` needs PyTorch + (optionally) CUDA. This binary uses
[`fastembed`](https://github.com/anush008/fastembed-rs), which runs the
**non-quantized fp32 BGE-M3 ONNX** export on CPU via ONNX Runtime — no Python, no
torch. Targets a cheap CPU VPS.

## Build / run

```bash
cargo build --release
./target/release/bge-m3-rs
```

First run downloads `BAAI/bge-m3` (~2.2 GB, cached afterwards) and the ONNX
Runtime shared lib (fetched by `ort` automatically). `cargo build` itself also
pulls the ONNX Runtime binary.

### Config (env, all optional — same names as `serve.py`)

| var | default | notes |
|-----|---------|-------|
| `BGE_M3_HOST` | `0.0.0.0` | |
| `BGE_M3_PORT` | `8000` | |
| `BGE_M3_BATCH_SIZE` | `32` | forward micro-batch |
| `BGE_M3_MAX_LENGTH` | `2048` | **must stay 2048** to match the index; fastembed's own default is 512 |
| `BGE_M3_INTRA_THREADS` | ort auto | set to #vCPU on a small box |

Then point the ingester/query side at it: `BGE_M3_URL=http://localhost:8000/embed`.

## Verify

```bash
# Self-tests (no data): dim, unit norm, determinism, related>unrelated cosine
cargo run --release --bin verify

# Real compatibility check against the index the Python service built.
# Export ONE row from LanceDB to ref.json: {"text": "<stored form>", "vector": [..1024..]}
cargo run --release --bin verify -- ref.json
```

The reference check prints `cosine(rust, stored)`. **Expect ~0.999** (fp16/ONNX
drift). `<0.99` means a *method* mismatch (pooling / max_length / quantization /
pre-normalization), not drift — investigate. Feed `text` already normalized the
way `embed.rs::normalize_for_embedding` produces (spans + tashkil stripped); the
tool embeds verbatim.

## Compatibility with the existing LanceDB

- Method matches `serve.py`: CLS token (`last_hidden_state[:, 0]`) + L2-normalize.
- The service embeds text **verbatim**. The Rust client (`embed.rs`) already
  strips spans + tashkil before calling; the server must not re-normalize.
- **Expected drift ~1e-3 element-wise**, *not* 1e-6: the index was built on GPU
  with fp16, and ONNX kernels differ from PyTorch's. This is retrieval-invisible
  (cosine drift ~1e-4) and not reducible on CPU against an fp16-built index.

## Hardware

Needs ~2.5 GB RAM resident → use a **≥4 GB** VPS. The int8 `BGEM3Q` variant fits
1 GB but deviates ~1e-2 from the stored vectors; not used here on purpose.

## systemd (VPS)

```ini
[Unit]
Description=bge-m3-rs embedding service
After=network.target

[Service]
ExecStart=/opt/bge-m3-rs/bge-m3-rs
Environment=BGE_M3_INTRA_THREADS=2
Restart=on-failure

[Install]
WantedBy=multi-user.target
```
