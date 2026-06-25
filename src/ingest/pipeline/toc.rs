//! Stage 3 — index `toc.jsonl` into a `page_id → chapter_path` map.
//!
//! Each TOC row is a heading on some `page_id`, linked to its parent heading via
//! `parent_id` (→ another row's `title_id`). We walk that chain to the root to
//! build the root→leaf title path, and key it by the heading's `page_id`.
//!
//! This index is a literal `page_id` lookup: only pages that start a heading
//! have an entry. The orchestrator (`process_book`) layers carry-forward on top,
//! so pages without their own heading inherit the most recent heading's path.

use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use crate::ingest::error::TocError;
use crate::ingest::model::TocRow;

/// Maps a page's `page_id` to its root→leaf chapter path.
#[derive(Debug, Default)]
pub struct ChapterIndex {
    by_page: HashMap<i64, Vec<String>>,
}

impl ChapterIndex {
    /// Chapter path for a page, or an empty slice if the page has no heading.
    pub fn path_for(&self, page_id: i64) -> Vec<String> {
        self.by_page.get(&page_id).cloned().unwrap_or_default()
    }

    pub fn len(&self) -> usize {
        self.by_page.len()
    }
}

#[tracing::instrument(skip_all, fields(book_id))]
pub fn load(book_dir: &Path) -> Result<ChapterIndex, TocError> {
    let path = book_dir.join("toc.jsonl");
    let reader = BufReader::new(File::open(path)?);

    // First pass: collect rows and index by title_id for parent lookups.
    let mut rows: Vec<TocRow> = Vec::new();
    for row in serde_json::Deserializer::from_reader(reader).into_iter::<TocRow>() {
        rows.push(row?);
    }
    let by_title: HashMap<i64, &TocRow> = rows.iter().map(|r| (r.title_id, r)).collect();

    // Second pass: walk each row's parent chain to its root, root→leaf.
    let mut by_page: HashMap<i64, Vec<String>> = HashMap::with_capacity(rows.len());
    for row in &rows {
        let path = walk_to_root(row, &by_title)?;
        // A page may carry several headings; keep the first one encountered.
        by_page.entry(row.page_id).or_insert(path);
    }

    let index = ChapterIndex { by_page };
    tracing::debug!(headings = rows.len(), pages_with_path = index.len(), "toc indexed");
    Ok(index)
}

/// Walk `row` up through `parent_id` links, returning titles root-first.
fn walk_to_root(row: &TocRow, by_title: &HashMap<i64, &TocRow>) -> Result<Vec<String>, TocError> {
    let mut leaf_to_root = vec![row.title_text.clone()];
    let mut seen = vec![row.title_id];
    let mut current = row.parent_id;

    while let Some(pid) = current {
        if seen.contains(&pid) {
            return Err(TocError::Cycle(pid));
        }
        let Some(parent) = by_title.get(&pid) else {
            // Dangling parent_id: stop the walk, keep what we have.
            break;
        };
        seen.push(pid);
        leaf_to_root.push(parent.title_text.clone());
        current = parent.parent_id;
    }

    leaf_to_root.reverse();
    Ok(leaf_to_root)
}
