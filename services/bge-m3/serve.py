"""BGE-M3 dense-embedding HTTP service for `shamela-ingest`.

Contract (must match `src/ingest/pipeline/embed.rs`):

    POST /embed   {"texts": ["...", ...]}  ->  {"embeddings": [[...; 1024], ...]}

One 1024-dim L2-normalized dense vector per input text, in the same order.
The Rust client validates the count and the dimensionality, so this service
returns exactly `len(texts)` vectors of width 1024.

Embeddings are computed with `transformers` directly — load BGE-M3 (an
XLM-RoBERTa _model), take the first/`[CLS]` token's last hidden state, and
L2-normalize. That is exactly BGE-M3's "dense" vector. We deliberately avoid
`FlagEmbedding` / `sentence-transformers` here: both import `datasets` ->
`pyarrow`, which segfaults on import in this environment, and neither is needed
just to serve dense embeddings.

Run (uv project; `uv run` syncs the env from pyproject.toml — incl. cu126 torch
— then serves):

    cd services/bge-m3
    uv run serve.py

Then point the ingester at it:

    BGE_M3_URL=http://localhost:8000/embed   # in the orchestrator .env

Config via environment variables (all optional):

    BGE_M3_MODEL        default "BAAI/bge-m3"
    BGE_M3_HOST         default "0.0.0.0"
    BGE_M3_PORT         default 8000
    BGE_M3_BATCH_SIZE   default 32     (forward micro-batch, to bound GPU memory)
    BGE_M3_MAX_LENGTH   default 2048   (chunks are <=600 whitespace tokens)
    BGE_M3_DEVICE       "cuda" | "cpu" (default: auto-detect)
"""

from __future__ import annotations

import logging
import os
import threading
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

EMBED_DIM = 1024  # BGE-M3 dense width; the Rust client requires exactly this.

MODEL_NAME = os.environ.get("BGE_M3_MODEL", "BAAI/bge-m3")
HOST = os.environ.get("BGE_M3_HOST", "0.0.0.0")
PORT = int(os.environ.get("BGE_M3_PORT", "8000"))
BATCH_SIZE = int(os.environ.get("BGE_M3_BATCH_SIZE", "32"))
MAX_LENGTH = int(os.environ.get("BGE_M3_MAX_LENGTH", "2048"))
DEVICE_OVERRIDE = os.environ.get("BGE_M3_DEVICE")  # None => auto-detect

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [bge-m3] %(message)s",
)
log = logging.getLogger("bge-m3")

# The _model is not safe to call concurrently; serialize encode() across requests.
# Each request is already a batch (the ingester sends ~32 texts), and the GPU is
# the bottleneck, so a single lock is the right amount of concurrency control.
_model = None
_tokenizer = None
_device = "cpu"
_lock = threading.Lock()


def _load_model() -> None:
    """Load BGE-M3 (tokenizer + _model) once at startup."""
    global _model, _tokenizer, _device
    import torch
    from transformers import AutoModel, AutoTokenizer

    if DEVICE_OVERRIDE:
        _device = DEVICE_OVERRIDE
    else:
        _device = "cuda" if torch.cuda.is_available() else "cpu"

    log.info("loading %s on %s ...", MODEL_NAME, _device)
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    _model = AutoModel.from_pretrained(MODEL_NAME).eval().to(_device)
    if _device == "cuda":
        _model.half()  # fp16 on GPU
    _model = _model
    log.info("_model ready (fp16=%s)", _device == "cuda")


def _encode(texts: list[str]) -> list[list[float]]:
    """CLS-pooled, L2-normalized dense vectors for `texts`, in order."""
    import torch

    out: list[list[float]] = []
    with torch.inference_mode():
        for start in range(0, len(texts), BATCH_SIZE):
            chunk = texts[start : start + BATCH_SIZE]
            batch = _tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            batch = {k: v.to(_device) for k, v in batch.items()}
            hidden = _model(**batch).last_hidden_state[:, 0]  # [CLS]/<s> token
            dense = torch.nn.functional.normalize(hidden, p=2, dim=1)
            out.extend(dense.float().cpu().tolist())
    return out


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _load_model()
    yield


app = FastAPI(title="BGE-M3 embedding service", version="1", lifespan=lifespan)


class EmbedRequest(BaseModel):
    texts: list[str] = Field(default_factory=list)


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if _model is not None else "loading",
        "_model": MODEL_NAME,
        "device": _device,
        "dim": EMBED_DIM,
    }


# Defined as a sync `def` so FastAPI runs it in a worker thread; the lock then
# serializes the actual _model call without blocking the event loop.
@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest) -> EmbedResponse:
    if _model is None:
        raise HTTPException(status_code=503, detail="_model still loading")
    if not req.texts:
        return EmbedResponse(embeddings=[])

    with _lock:
        embeddings = _encode(req.texts)

    if embeddings and len(embeddings[0]) != EMBED_DIM:
        raise HTTPException(
            status_code=500,
            detail=f"_model produced {len(embeddings[0])}-dim vectors, expected {EMBED_DIM}",
        )
    return EmbedResponse(embeddings=embeddings)


if __name__ == "__main__":
    # Single worker: the _model is loaded once per process and guarded by a lock.
    uvicorn.run(app, host=HOST, port=PORT, workers=1, log_level="info")
