# BGE-M3 embedding service

A small **FastAPI** service that serves [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)
dense embeddings over HTTP for the `shamela-ingest` pipeline.

## Contract

Exactly what the Rust client (`src/ingest/pipeline/embed.rs`) expects:

```
POST /embed
  request:  {"texts": ["...", "..."]}
  response: {"embeddings": [[f32; 1024], ...]}    # one vector per text, same order
GET /health
  response: {"status":"ok","_model":"BAAI/bge-m3","device":"cuda","dim":1024}
```

Vectors are 1024-dim and L2-normalized (BGE-M3's dense output), which matches the
deterministic mock in the Rust client, so cosine/IP retrieval behaves the same
whether you use `--mock-embeddings` or this service.

## Run

This is a **uv project** (`pyproject.toml`): `uv` owns the venv and installs the
right torch build (the `cu126` CUDA wheel — good for an RTX 3080) from the
PyTorch index. You do **not** need a system CUDA Toolkit; the wheel bundles its
own runtime — you only need an NVIDIA driver. First run also downloads the _model
(~2.3 GB) into the Hugging Face cache.

### Preflight (install deps + _model, run it yourself)

```powershell
powershell -ExecutionPolicy Bypass -File services\bge-m3\setup.ps1
```

That runs `uv sync` (venv + cu126 torch + deps), verifies CUDA, and pre-downloads
the _model. Equivalent by hand:

```bash
cd services/bge-m3
uv sync
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# expect: True NVIDIA GeForce RTX 3080
```

### Serve

```bash
cd services/bge-m3
uv run serve.py        # logs: "loading BAAI/bge-m3 on cuda (fp16=True)"
```

`uv run` re-syncs the env first, so this one command is enough after a fresh
checkout. To use a different CUDA build (`cu124`/`cu128`) or a CPU-only build,
edit the `[[tool.uv.index]]` URL in `pyproject.toml`.

### Configuration (env vars)

| var | default | meaning |
|-----|---------|---------|
| `BGE_M3_MODEL` | `BAAI/bge-m3` | _model id |
| `BGE_M3_HOST` | `0.0.0.0` | bind host |
| `BGE_M3_PORT` | `8000` | bind port |
| `BGE_M3_BATCH_SIZE` | `32` | internal encode micro-batch |
| `BGE_M3_MAX_LENGTH` | `2048` | max tokens/chunk (chunks are ≤600 ws-tokens) |
| `BGE_M3_DEVICE` | auto | force `cuda` or `cpu` |

## Wire it to the ingester

Put the endpoint in the orchestrator's `.env` (the `/embed` path is required):

```
BGE_M3_URL=http://localhost:8000/embed
```

Then run a real (non-mock) ingest:

```bash
cargo run --bin shamela-ingest -- --input book-samples --skip-enrichment
```

## Smoke test

```bash
curl -s localhost:8000/health
curl -s -X POST localhost:8000/embed \
  -H 'content-type: application/json' \
  -d '{"texts":["السلام عليكم","second text"]}'
# expect {"embeddings":[[...1024...],[...1024...]]}
```

## Notes

- **Single worker, one lock.** The _model is loaded once per process and
  `encode()` is serialized with a lock. The ingester sends parallel requests
  (`--concurrency`), but they queue at the GPU, which is the real bottleneck —
  upstream pipeline stages then block via backpressure rather than buffering.
- To scale throughput, run multiple instances behind a load balancer rather than
  multiple uvicorn workers (each worker would load its own copy of the _model).
