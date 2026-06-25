//! Stage 2 — load `book_metadata.json`, the bibliographic record that supplies
//! every chunk's denormalized book-level columns.

use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use crate::ingest::error::BookMetaError;
use crate::ingest::model::BookMetadata;

#[tracing::instrument(skip_all, fields(book_id))]
pub fn load(book_dir: &Path) -> Result<BookMetadata, BookMetaError> {
    let path = book_dir.join("book_metadata.json");
    let meta: BookMetadata = serde_json::from_reader(BufReader::new(File::open(path)?))?;
    tracing::Span::current().record("book_id", meta.book_id);
    Ok(meta)
}
