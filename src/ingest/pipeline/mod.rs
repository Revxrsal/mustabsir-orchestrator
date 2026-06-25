//! Orchestrator: discover books, run them concurrently (capped by
//! `--concurrency`), and funnel all writes through a single LanceDB writer task.
//!
//! Concurrency model (and a deliberate simplification, flagged to the user):
//!   * Across books: a `Semaphore` + `JoinSet` cap in-flight books. Each book is
//!     an independent task returning `Result<BookSummary, BookError>`; a failure
//!     is logged + appended to `failures.jsonl` and never crashes the run.
//!   * Within a book: a streaming loop `stream → clean → chunk → batch(32) →
//!     embed → RecordBatch`, then the batch is sent to the writer. Awaiting
//!     `embed` (and the bounded writer channel) is what enforces backpressure —
//!     a book cannot buffer ahead of the embedder. The spec sketches literal
//!     per-stage channels inside a book; that adds CPU/IO overlap whose only
//!     payoff appears with the real (milestone-4) embedder, so it is deferred.
//!   * Writes from all books go through ONE writer task. Concurrent
//!     `Table::add` would race on Lance commits; a single writer serializes them.

mod book;
mod chunk;
pub mod embed;
mod enrich;
mod manifest;
mod pages;
mod toc;
mod write;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::record_batch::RecordBatch;
use dotenv_codegen::dotenv;
use lancedb::index::scalar::FtsIndexBuilder;
use lancedb::index::Index;
use lancedb::Table;
use rig::providers::anthropic::Client as AnthropicClient;
use serde::Serialize;
use tokio::sync::{Semaphore, mpsc};
use tokio::task::JoinSet;

use crate::ingest::Cli;
use crate::ingest::error::{BookError, WriteError};
use crate::ingest::model::{BookEnrichment, BookSummary, Chunk, Era};
use embed::Embedder;

/// Embedding batch size (chunks per BGE-M3 request).
const BATCH_SIZE: usize = 32;
/// Bounded depth of the channel feeding the single writer task.
const WRITER_CHANNEL_DEPTH: usize = 8;
/// Whitespace tokens of each page carried onto the front of the next page, so a
/// passage straddling a page break lands whole in a chunk on each side. Pages in
/// this corpus are tiny (median ~25 tokens), so without this the boundary
/// between two pages is never co-located in any single chunk.
const PAGE_OVERLAP: usize = 40;

#[derive(Serialize)]
struct FailureRecord {
    book_dir: String,
    error: String,
}

/// Run the full ingestion.
pub(crate) async fn run(cli: &Cli) -> anyhow::Result<()> {
    // Load `.env` once so both ANTHROPIC_API_KEY and BGE_M3_URL are available
    // regardless of which stages run.
    dotenv::dotenv().ok();

    let books = discover_books(&cli.input)?;
    if books.is_empty() {
        anyhow::bail!("no book directories found under {}", cli.input.display());
    }
    tracing::info!(count = books.len(), "discovered books");

    if cli.dry_run {
        tracing::info!(
            count = books.len(),
            input = %cli.input.display(),
            "dry run: discovery only, exiting without ingesting",
        );
        return Ok(());
    }

    let embedder = Arc::new(Embedder::from_config(cli.mock_embeddings)?);

    // The Anthropic client is only built when enrichment will actually run, so
    // `--skip-enrichment` works without an API key.
    let anthropic = if cli.skip_enrichment {
        None
    } else {
        let key = dotenv!("ANTHROPIC_API_KEY");
        Some(Arc::new(AnthropicClient::builder().api_key(key).build()?))
    };
    let cache_root = PathBuf::from("cache/enrichments");

    let db_path = cli.lancedb.to_string_lossy().to_string();
    let conn = write::connect(&db_path).await?;
    let table = write::open_or_create_table(&conn, &cli.table).await?;
    // if !cli.dry_run {
    //     tracing::info!(table = %cli.table, "truncating existing records for a clean ingestion run");
    //     // LanceDB allows an SQL expression string to filter what to delete.
    //     // "1=1" matches all rows, clearing the table completely while keeping the schema intact.
    //     table.delete("1=1").await?;
    // }
    // Single writer task; all per-book tasks send batches here.
    let (tx, rx) = mpsc::channel::<RecordBatch>(WRITER_CHANNEL_DEPTH);
    let writer = tokio::spawn(writer_task(table, rx));

    // Cap in-flight books.
    let sem = Arc::new(Semaphore::new(cli.concurrency.max(1)));
    let mut set: JoinSet<(PathBuf, Result<BookSummary, BookError>)> = JoinSet::new();
    for dir in books {
        let permit = sem.clone().acquire_owned().await.expect("semaphore open");
        let embedder = embedder.clone();
        let tx = tx.clone();
        let skip_enrichment = cli.skip_enrichment;
        let anthropic = anthropic.clone();
        let cache_root = cache_root.clone();
        set.spawn(async move {
            let _permit = permit;
            let res =
                process_book(dir.clone(), embedder, tx, skip_enrichment, anthropic, cache_root)
                    .await;
            (dir, res)
        });
    }
    drop(tx); // only the per-book tasks hold senders now

    let mut summaries: Vec<BookSummary> = Vec::new();
    let mut failures: Vec<FailureRecord> = Vec::new();
    while let Some(joined) = set.join_next().await {
        match joined {
            Ok((_dir, Ok(summary))) => summaries.push(summary),
            Ok((dir, Err(err))) => {
                tracing::error!(book_dir = %dir.display(), error = %err, "book failed");
                failures.push(FailureRecord {
                    book_dir: dir.display().to_string(),
                    error: err.to_string(),
                });
            }
            Err(join_err) => {
                tracing::error!(error = %join_err, "book task panicked");
                failures.push(FailureRecord {
                    book_dir: "<panicked task>".to_string(),
                    error: join_err.to_string(),
                });
            }
        }
    }

    let rows_written = writer
        .await
        .map_err(|e| anyhow::anyhow!("writer task panicked: {e}"))??;

    // Build the BM25 full-text index on the normalized column so hybrid
    // (dense + lexical) search works. Done once here, after all rows are written.
    if rows_written > 0 {
        let table = conn.open_table(&cli.table).execute().await?;
        table
            .create_index(&["text_normalized"], Index::FTS(FtsIndexBuilder::default()))
            .replace(true)
            .execute()
            .await?;
        tracing::info!("built FTS index on text_normalized");
    }

    if !failures.is_empty() {
        append_failures(&failures)?;
    }

    let total_chunks: usize = summaries.iter().map(|s| s.chunk_count).sum();
    tracing::info!(
        books_ok = summaries.len(),
        books_failed = failures.len(),
        chunks = total_chunks,
        rows_written,
        "ingest complete",
    );
    for s in &summaries {
        tracing::info!(
            book_id = s.book_id,
            title = %s.title_ar,
            pages = s.page_count,
            chunks = s.chunk_count,
            "book summary",
        );
    }

    Ok(())
}

/// The single writer: drains batches and appends them to the table in order.
async fn writer_task(table: Table, mut rx: mpsc::Receiver<RecordBatch>) -> Result<usize, WriteError> {
    let mut total = 0usize;
    while let Some(batch) = rx.recv().await {
        total += batch.num_rows();
        write::add_batch(&table, batch).await?;
    }
    Ok(total)
}

/// Process one book end-to-end into batches sent to the writer.
#[tracing::instrument(skip_all, fields(book_dir = %dir.display(), book_id))]
async fn process_book(
    dir: PathBuf,
    embedder: Arc<Embedder>,
    writer_tx: mpsc::Sender<RecordBatch>,
    skip_enrichment: bool,
    anthropic: Option<Arc<AnthropicClient>>,
    cache_root: PathBuf,
) -> Result<BookSummary, BookError> {
    manifest::load(&dir)?;
    let meta = book::load(&dir)?;
    tracing::Span::current().record("book_id", meta.book_id);

    if meta.is_hidden {
        tracing::warn!(book_id = meta.book_id, "skipping hidden book (is_hidden=true)");
        return Ok(BookSummary {
            book_id: meta.book_id,
            title_ar: meta.title_ar,
            chunk_count: 0,
            page_count: 0,
            enrichment_cached: false,
        });
    }

    let chapters = toc::load(&dir)?;

    let mut enrichment_cached = false;
    let enrichment = if skip_enrichment {
        BookEnrichment::skipped()
    } else {
        let client = anthropic
            .as_ref()
            .expect("anthropic client is present when enrichment is not skipped");
        let first_body = pages::first_page(&dir)?.map(|p| p.body);
        let (e, cached) = enrich::enrich(&meta, first_body.as_deref(), client, &cache_root).await?;
        enrichment_cached = cached;
        e
    };

    let era = Era::from_death_hijri(meta.main_author_death_hijri);

    let mut buf: Vec<Chunk> = Vec::with_capacity(BATCH_SIZE);
    let mut chunk_count = 0usize;
    let mut page_count = 0usize;
    // Carry-forward chapter: pages without their own heading inherit the path of
    // the most recent preceding heading page. Relies on pages.jsonl being in
    // sequence order (verified for the sample corpus).
    let mut current_chapter: Vec<String> = Vec::new();
    // Tail of the previous page, prepended to the next for inter-page overlap.
    let mut prev_tail: String = String::new();

    for page_res in pages::stream(&dir)? {
        let page = page_res?;
        page_count += 1;
        let cleaned = pages::clean(&page.body);
        let path = chapters.path_for(page.page_id);
        if !path.is_empty() {
            current_chapter = path;
        }
        // Prepend the previous page's tail so a passage crossing the page break
        // is chunked together. The tail for the *next* page is taken from this
        // page's own cleaned body (not the overlapped text), so overlap never
        // cascades beyond one boundary.
        let with_overlap = if prev_tail.is_empty() {
            cleaned.clone()
        } else {
            format!("{prev_tail} {cleaned}")
        };
        prev_tail = chunk::tail_tokens(&cleaned, PAGE_OVERLAP);

        let chunks =
            chunk::chunk_page(&with_overlap, &page, &meta, &enrichment, era, &current_chapter);
        for c in chunks {
            buf.push(c);
            if buf.len() >= BATCH_SIZE {
                chunk_count += flush(&mut buf, &embedder, &writer_tx).await?;
            }
        }
    }
    if !buf.is_empty() {
        chunk_count += flush(&mut buf, &embedder, &writer_tx).await?;
    }

    Ok(BookSummary {
        book_id: meta.book_id,
        title_ar: meta.title_ar,
        chunk_count,
        page_count,
        enrichment_cached,
    })
}

/// Build the string actually sent to the embedder for a chunk: the root→leaf
/// chapter path prepended as a context header, then the chunk body. The header
/// is **not** stored — the `text` column keeps the clean body — it only steers
/// the vector by anchoring the chunk in its section's topical region (so e.g. a
/// قصر-الصلاة chunk lands near "Prayer", not near ambiguous Fasting/Umrah text).
/// Diacritic/span normalization happens downstream in `Embedder::embed`.
fn embed_text(c: &Chunk) -> String {
    if c.chapter_path.is_empty() {
        c.text.clone()
    } else {
        format!("{} -> {}", c.chapter_path.join(" -> "), c.text)
    }
}

/// Embed the buffered chunks, build a batch, and hand it to the writer.
async fn flush(
    buf: &mut Vec<Chunk>,
    embedder: &Embedder,
    writer_tx: &mpsc::Sender<RecordBatch>,
) -> Result<usize, BookError> {
    let texts: Vec<String> = buf.iter().map(embed_text).collect();
    let embeddings = embedder.embed(&texts).await?;
    for (c, e) in buf.iter_mut().zip(embeddings) {
        c.embedding = e;
    }
    let n = buf.len();
    let batch = write::chunks_to_batch(buf.as_slice())?;
    writer_tx
        .send(batch)
        .await
        .map_err(|_| BookError::Write(WriteError::WriterClosed))?;
    buf.clear();
    Ok(n)
}

/// Recursively find book directories under `input`. A directory containing
/// `book_metadata.json` is treated as a book (and not descended into). This
/// handles the flat sample layout (`book-samples/bookN/`), a single category,
/// and the real DB's nested `<category>/<book>/` layout — including when
/// `input` itself is one book directory.
fn discover_books(input: &Path) -> anyhow::Result<Vec<PathBuf>> {
    let mut dirs = Vec::new();
    collect_books(input, &mut dirs)?;
    dirs.sort();
    Ok(dirs)
}

fn collect_books(dir: &Path, out: &mut Vec<PathBuf>) -> anyhow::Result<()> {
    if dir.join("book_metadata.json").exists() {
        out.push(dir.to_path_buf());
        return Ok(());
    }
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() {
            collect_books(&path, out)?;
        }
    }
    Ok(())
}

/// Append per-book failures to `failures.jsonl` (one JSON object per line).
fn append_failures(failures: &[FailureRecord]) -> anyhow::Result<()> {
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open("failures.jsonl")?;
    for record in failures {
        writeln!(file, "{}", serde_json::to_string(record)?)?;
    }
    tracing::warn!(count = failures.len(), "appended to failures.jsonl");
    Ok(())
}
