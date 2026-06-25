//! Thin binary entry for the Shamela ingestion pipeline.
//!
//! The implementation lives in the `ingest` module at `src/ingest/`, shared in
//! place via `#[path]` so it stays a normal module of the project (per the
//! agreed layout) without needing a separate library crate.

#[path = "../ingest/mod.rs"]
mod ingest;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    ingest::run().await
}
