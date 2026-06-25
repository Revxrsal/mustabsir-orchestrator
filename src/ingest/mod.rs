//! `shamela-ingest` — walks a directory of Shamela book exports and writes
//! per-chunk records (with BGE-M3 embeddings) to a LanceDB table.
//!
//! Pipeline (per book): manifest load → metadata load → TOC index →
//! enrichment (Claude) → stream pages → clean → chunk → batch → embed → write.
//!
//! This module is the entry point: [`run`] parses the CLI, sets up tracing, and
//! hands off to [`pipeline::run`]. It is driven by the `shamela-ingest` binary
//! (`src/bin/shamela-ingest.rs`).
//!
//! Milestone 2 status: `manifest`, `book`, `toc`, `pages`, `chunk`, `embed`
//! (mock path), and `write` are implemented. `enrich` requires
//! `--skip-enrichment`; the real embedding endpoint and resume-state come later.

pub mod error;
pub mod model;
pub mod pipeline;

use std::path::PathBuf;

use clap::Parser;

/// Ingest Shamela book exports into a LanceDB table.
#[derive(Debug, Parser)]
#[command(name = "shamela-ingest", version, about)]
pub(crate) struct Cli {
    /// Directory containing one sub-directory per book (each with manifest.json,
    /// book_metadata.json, toc.jsonl, pages.jsonl).
    #[arg(long, default_value = "book-samples")]
    pub input: PathBuf,

    /// Maximum number of books processed concurrently (JoinSet cap).
    #[arg(long, default_value_t = 4)]
    pub concurrency: usize,

    /// LanceDB database directory. The table `--table` is created/opened inside it.
    #[arg(long, default_value = "lancedb")]
    pub lancedb: PathBuf,

    /// LanceDB table name (single table, no partitioning).
    #[arg(long, default_value = "chunks")]
    pub table: String,

    /// Use deterministic pseudo-random unit vectors instead of calling BGE-M3.
    /// Lets the full pipeline run before the GPU embedding service exists.
    #[arg(long)]
    pub mock_embeddings: bool,

    /// Discover books (recursively) and report the count, then exit without
    /// touching LanceDB, the embedder, or the Anthropic client. Useful for
    /// sizing a large corpus before a real run.
    #[arg(long)]
    pub dry_run: bool,

    /// Skip the Claude enrichment stage; madhhab/aqeedah left `None`,
    /// sect `unknown`, canonicity_tier 3 (neutral), is_critical_edition false.
    #[arg(long)]
    pub skip_enrichment: bool,
}

/// CLI entry point for the ingestion pipeline.
pub async fn run() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                // Lance is extremely chatty at INFO (per-commit/file events); cap it.
                .unwrap_or_else(|_| "info,shamela_ingest=debug,lance=warn,lancedb=warn".into()),
        )
        .init();

    let cli = Cli::parse();
    pipeline::run(&cli).await
}
