//! Stage 1 — load `manifest.json` and confirm the files it lists are present.
//!
//! Checksum verification was intentionally dropped, so this is a light gate: it
//! catches a missing/corrupt manifest or a book directory that is missing one of
//! the files the manifest references, and surfaces the page/toc counts for logs.

use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use crate::ingest::error::ManifestError;
use crate::ingest::model::Manifest;

/// Load and parse `manifest.json`, then verify every file it lists exists in the
/// book directory.
#[tracing::instrument(skip_all, fields(book_id))]
pub fn load(book_dir: &Path) -> Result<Manifest, ManifestError> {
    let path = book_dir.join("manifest.json");
    let file = File::open(&path).map_err(|source| ManifestError::Io {
        path: path.display().to_string(),
        source,
    })?;
    let manifest: Manifest = serde_json::from_reader(BufReader::new(file))?;

    tracing::Span::current().record("book_id", manifest.book_id);

    for entry in &manifest.files {
        if !book_dir.join(&entry.path).exists() {
            return Err(ManifestError::MissingFile {
                file: entry.path.clone(),
            });
        }
    }

    tracing::debug!(
        pages = manifest.page_count,
        toc = manifest.toc_count,
        "manifest loaded",
    );
    Ok(manifest)
}
