//! BGE-M3 dense-embedding HTTP service — CPU-only Rust port of `serve.py`.
//!
//! Contract (must match `src/ingest/pipeline/embed.rs`, identical to the Python
//! service it replaces):
//!
//!     POST /embed   {"texts": ["...", ...]}  ->  {"embeddings": [[...; 1024], ...]}
//!
//! One 1024-dim L2-normalized dense vector per input text, in the same order.
//!
//! How this stays compatible with the LanceDB built by `serve.py`:
//!   * fastembed's `EmbeddingModel::BGEM3` is the *non-quantized* fp32 ONNX
//!     export, CLS-pooled and L2-normalized — the same dense vector the Python
//!     service computed as `last_hidden_state[:, 0]` then `normalize(p=2)`.
//!   * `MAX_LENGTH` defaults to 2048, matching `serve.py` (fastembed's own
//!     default is 512, which would truncate long chunks differently — so we set
//!     it explicitly).
//!   * The service embeds text **verbatim**. The Rust client already applies
//!     `normalize_for_embedding` (strip spans + tashkil) before calling, exactly
//!     as it did at ingest; the server must not re-normalize.
//!
//! Expected drift vs the stored vectors is ~1e-3 element-wise — the index was
//! built on GPU with fp16 (`serve.py` `_model.half()`), and ONNX kernels differ
//! from PyTorch's. That is retrieval-invisible (cosine drift ~1e-4); it is *not*
//! reducible to 1e-6 by any CPU implementation against an fp16-built index.
//!
//! Config via environment variables (all optional), mirroring `serve.py`:
//!
//!     BGE_M3_HOST           default "0.0.0.0"
//!     BGE_M3_PORT           default 8000
//!     BGE_M3_BATCH_SIZE     default 32
//!     BGE_M3_MAX_LENGTH     default 2048
//!     BGE_M3_INTRA_THREADS  default: ort auto (set to #vCPU on a small VPS)

use std::env;
use std::sync::{Arc, Mutex};

use axum::{
    extract::State,
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};
use serde::{Deserialize, Serialize};

/// BGE-M3 dense width; the Rust client requires exactly this.
const EMBED_DIM: usize = 1024;

const DEFAULT_HOST: &str = "0.0.0.0";
const DEFAULT_PORT: u16 = 8000;
const DEFAULT_BATCH_SIZE: usize = 32;
const DEFAULT_MAX_LENGTH: usize = 2048;

#[derive(Clone)]
struct AppState {
    /// `TextEmbedding::embed` takes `&mut self`, so requests are serialized
    /// behind a `Mutex` — exactly the single-lock model `serve.py` used. Each
    /// request is already a batch and ort parallelizes internally, so this is
    /// the right amount of concurrency control.
    model: Arc<Mutex<TextEmbedding>>,
    batch_size: usize,
}

#[derive(Deserialize)]
struct EmbedRequest {
    #[serde(default)]
    texts: Vec<String>,
}

#[derive(Serialize)]
struct EmbedResponse {
    embeddings: Vec<Vec<f32>>,
}

#[derive(Serialize)]
struct HealthResponse {
    status: &'static str,
    model: &'static str,
    device: &'static str,
    dim: usize,
}

fn env_parse<T: std::str::FromStr>(key: &str, default: T) -> T {
    env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .with_target(false)
        .init();

    let host = env::var("BGE_M3_HOST").unwrap_or_else(|_| DEFAULT_HOST.to_string());
    let port: u16 = env_parse("BGE_M3_PORT", DEFAULT_PORT);
    let batch_size: usize = env_parse("BGE_M3_BATCH_SIZE", DEFAULT_BATCH_SIZE);
    let max_length: usize = env_parse("BGE_M3_MAX_LENGTH", DEFAULT_MAX_LENGTH);

    tracing::info!("loading BAAI/bge-m3 (fp32 ONNX, CPU) max_length={max_length} ...");

    // Loading downloads the model on first run, then caches it (~2.2 GB on disk,
    // ~2–2.5 GB resident). Run on a blocking thread so we don't stall the async
    // runtime during the (slow) first-time download + session init.
    let model = tokio::task::spawn_blocking(move || -> anyhow::Result<TextEmbedding> {
        let mut opts = TextInitOptions::new(EmbeddingModel::BGEM3)
            .with_max_length(max_length)
            .with_show_download_progress(true);
        if let Ok(n) = env::var("BGE_M3_INTRA_THREADS") {
            if let Ok(n) = n.parse::<usize>() {
                opts = opts.with_intra_threads(n);
            }
        }
        Ok(TextEmbedding::try_new(opts)?)
    })
    .await??;

    tracing::info!("model ready");

    let state = AppState { model: Arc::new(Mutex::new(model)), batch_size };

    let app = Router::new()
        .route("/health", get(health))
        .route("/embed", post(embed))
        .with_state(state);

    let addr = format!("{host}:{port}");
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    tracing::info!("listening on http://{addr}");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        model: "BAAI/bge-m3",
        device: "cpu",
        dim: EMBED_DIM,
    })
}

/// Embed a batch. `embed()` is CPU-bound and needs `&mut self`, so it runs on a
/// blocking thread under the `Mutex`, serializing concurrent requests.
async fn embed(
    State(state): State<AppState>,
    Json(req): Json<EmbedRequest>,
) -> Result<Json<EmbedResponse>, (StatusCode, String)> {
    if req.texts.is_empty() {
        return Ok(Json(EmbedResponse { embeddings: vec![] }));
    }

    let model = state.model.clone();
    let batch_size = state.batch_size;
    let texts = req.texts;

    let embeddings = tokio::task::spawn_blocking(move || {
        let mut model = model.lock().expect("embed mutex poisoned");
        model.embed(texts, Some(batch_size))
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("join error: {e}")))?
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("embed failed: {e}")))?;

    if let Some(bad) = embeddings.iter().find(|v| v.len() != EMBED_DIM) {
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("model produced {}-dim vectors, expected {EMBED_DIM}", bad.len()),
        ));
    }

    Ok(Json(EmbedResponse { embeddings }))
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
    tracing::info!("shutting down");
}
