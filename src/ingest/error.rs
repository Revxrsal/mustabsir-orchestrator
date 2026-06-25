//! Typed, per-stage errors. `anyhow` is reserved for `main` and the per-book
//! task boundary; everything inside a stage returns one of these `thiserror`
//! enums. `BookError` is the per-book aggregate (`Result<BookSummary, BookError>`
//! from each task); a failing book appends to `failures.jsonl` and the run
//! continues — it never propagates up and crashes the orchestrator.
//!
//! `EnrichError`/`StateError` (and a few of their variants) are defined ahead of
//! the milestones that consume them, so unused variants are allowed for now.

#![allow(dead_code)]

use thiserror::Error;

/// Stage 1 — `manifest.json` load + presence check. Checksum verification was
/// intentionally dropped, so this only fails on a missing/unparseable manifest
/// or a file the manifest lists that isn't on disk.
#[derive(Debug, Error)]
pub enum ManifestError {
    #[error("reading {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("parsing manifest.json: {0}")]
    Parse(#[from] serde_json::Error),
    #[error("manifest lists {file} but it is missing from the book directory")]
    MissingFile { file: String },
}

/// Stage 2 — `book_metadata.json` load.
#[derive(Debug, Error)]
pub enum BookMetaError {
    #[error("reading book_metadata.json: {0}")]
    Io(#[from] std::io::Error),
    #[error("parsing book_metadata.json: {0}")]
    Parse(#[from] serde_json::Error),
}

/// Stage 3 — TOC indexing into per-page chapter paths.
#[derive(Debug, Error)]
pub enum TocError {
    #[error("reading toc.jsonl: {0}")]
    Io(#[from] std::io::Error),
    #[error("parsing toc.jsonl row: {0}")]
    Parse(#[from] serde_json::Error),
    #[error("parent_id chain cycle detected at title_id {0}")]
    Cycle(i64),
}

/// Stage 4 — page streaming + cleaning.
#[derive(Debug, Error)]
pub enum PageError {
    #[error("reading pages.jsonl: {0}")]
    Io(#[from] std::io::Error),
    #[error("parsing pages.jsonl row: {0}")]
    Parse(#[from] serde_json::Error),
}

/// Stage 5 — book-level enrichment via rig + the JSON cache.
#[derive(Debug, Error)]
pub enum EnrichError {
    #[error("enrichment cache I/O: {0}")]
    Cache(#[from] std::io::Error),
    #[error("(de)serializing cached enrichment: {0}")]
    CacheParse(#[from] serde_json::Error),
    /// Wraps `rig` extractor/provider failures (typed in the enrich milestone).
    #[error("enrichment LLM call failed: {0}")]
    Llm(String),
}

/// Stage 6 — BGE-M3 HTTP embedding.
#[derive(Debug, Error)]
pub enum EmbedError {
    #[error("BGE-M3 request failed: {0}")]
    Http(#[from] reqwest::Error),
    #[error("BGE-M3 returned {got} embeddings for {sent} texts")]
    CountMismatch { sent: usize, got: usize },
    #[error("BGE-M3 returned a {got}-dim vector, expected {expected}")]
    DimMismatch { expected: usize, got: usize },
    #[error("BGE_M3_URL not set and --mock-embeddings not passed")]
    NoEndpoint,
}

/// Stage 7 — RecordBatch build + `lance::Table::add`.
#[derive(Debug, Error)]
pub enum WriteError {
    #[error("Arrow RecordBatch build failed: {0}")]
    Arrow(#[from] arrow::error::ArrowError),
    #[error("LanceDB error: {0}")]
    Lance(#[from] lancedb::Error),
    #[error("chunk has a {got}-dim embedding, expected {expected}")]
    EmbeddingDim { expected: usize, got: usize },
    #[error("writer task closed before all batches were written")]
    WriterClosed,
}

/// Resume-state store (`rusqlite`).
#[derive(Debug, Error)]
pub enum StateError {
    /// Wraps `rusqlite` failures (typed in the state milestone).
    #[error("resume-state store failed: {0}")]
    Backend(String),
}

/// Per-book aggregate. Each per-book task returns `Result<BookSummary, BookError>`;
/// the `book_id` is attached so failures land in `failures.jsonl` with context.
#[derive(Debug, Error)]
pub enum BookError {
    #[error("manifest: {0}")]
    Manifest(#[from] ManifestError),
    #[error("book metadata: {0}")]
    BookMeta(#[from] BookMetaError),
    #[error("toc: {0}")]
    Toc(#[from] TocError),
    #[error("pages: {0}")]
    Page(#[from] PageError),
    #[error("enrich: {0}")]
    Enrich(#[from] EnrichError),
    #[error("embed: {0}")]
    Embed(#[from] EmbedError),
    #[error("write: {0}")]
    Write(#[from] WriteError),
    #[error("state: {0}")]
    State(#[from] StateError),
    #[error("book directory malformed: {0}")]
    Malformed(String),
}
