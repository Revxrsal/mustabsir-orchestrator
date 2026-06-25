//! Data structures for the Shamela ingestion pipeline.
//!
//! Three layers live here:
//!   1. Raw on-disk records (`Manifest`, `BookMetadata`, `TocRow`, `PageRow`) —
//!      `serde::Deserialize` mirrors of the four files in each book directory.
//!   2. Enrichment I/O (`BookEnrichment` + its enums) — the `rig` Extractor
//!      output, also serialized into the per-book JSON cache.
//!   3. The LanceDB row contract (`Chunk`) and per-book result (`BookSummary`).
//!
//! The only behaviour here is the mechanical `Era` mapping. Several raw fields
//! are parsed but not yet consumed downstream, so dead code is allowed in this
//! module.

#![allow(dead_code)]

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Sentinel value Shamela uses in `main_author_death_hijri` for
/// "unknown / author still alive". Mapped to `Era::Modern`, never skipped.
pub const DEATH_HIJRI_UNKNOWN: i32 = 99999;

// ---------------------------------------------------------------------------
// 1. Raw on-disk records
// ---------------------------------------------------------------------------

/// `manifest.json` — checksums + counts used to verify a book directory.
#[derive(Debug, Clone, Deserialize)]
pub struct Manifest {
    pub book_id: i64,
    pub shamela_id: i64,
    pub book_type: i32,
    pub book_type_label: String,
    pub title_ar: String,
    pub page_count: i64,
    pub toc_count: i64,
    pub files: Vec<FileEntry>,
    pub extracted_at: String,
    pub extractor_version: String,
    pub snapshot_id: String,
    /// Free-form reject records; shape is not relied upon.
    #[serde(default)]
    pub rejects: Vec<serde_json::Value>,
}

/// One entry in `manifest.files`: the file to hash and its expected digest.
#[derive(Debug, Clone, Deserialize)]
pub struct FileEntry {
    pub path: String,
    pub sha256: String,
    pub bytes: u64,
    pub rows: u64,
}

/// `book_metadata.json` — the bibliographic record. Source of the denormalized
/// book-level columns on every `Chunk`.
#[derive(Debug, Clone, Deserialize)]
pub struct BookMetadata {
    pub book_id: i64,
    pub shamela_id: i64,
    pub title_ar: String,
    pub book_type: i32,
    pub book_type_label: String,
    pub category_id: i32,
    pub category_name_ar: String,
    pub main_author_id: i64,
    pub main_author_name_ar: String,
    /// May be the `99999` sentinel ([`DEATH_HIJRI_UNKNOWN`]).
    pub main_author_death_hijri: i32,
    pub main_author_death_hijri_text: Option<String>,
    pub authors_text: String,
    pub authors: Vec<Author>,
    pub hijri_era: i64,
    pub printed: bool,
    /// When `true` the book is skipped (with a logged note), per project decision.
    pub is_hidden: bool,
    pub parent_id: Option<i64>,
    pub group_id: Option<i64>,
    pub version_major: i32,
    pub version_minor: i32,
    /// The Arabic "betaka" blurb; fed into the enrichment prompt.
    pub betaka_text: String,
    /// `{"date": "DDMMYYYY"}` in samples; shape not relied upon.
    #[serde(default)]
    pub meta: serde_json::Value,
    pub volume_count_observed: i32,
    pub has_multi_part: bool,
}

/// One author in `book_metadata.authors`.
#[derive(Debug, Clone, Deserialize)]
pub struct Author {
    pub author_id: i64,
    pub role: String,
    pub name_ar: String,
    pub death_hijri: i32,
    pub death_hijri_text: Option<String>,
}

/// One row of `toc.jsonl` — a heading. `parent_id` links to another `title_id`,
/// forming the hierarchy walked into a `Chunk::chapter_path`.
#[derive(Debug, Clone, Deserialize)]
pub struct TocRow {
    pub title_id: i64,
    pub book_id: i64,
    pub page_id: i64,
    pub parent_id: Option<i64>,
    pub shamela_title_id: i64,
    pub title_text: String,
}

/// One row of `pages.jsonl` — a page body, streamed one at a time.
#[derive(Debug, Clone, Deserialize)]
pub struct PageRow {
    pub page_id: i64,
    pub book_id: i64,
    pub shamela_page_id: i64,
    /// Volume/part label, e.g. `"1".."7"` in multi-part books; `null` otherwise.
    pub part: Option<String>,
    /// Printed page number. `null` in the source for pages with no printed
    /// number (front matter, etc.) — kept `Option` so those rows still parse;
    /// coerced to the `-1` sentinel when built into a `Chunk` (see `chunk.rs`),
    /// which keeps the LanceDB `page_num` column non-nullable Int32 as written.
    pub page_num: Option<i32>,
    pub sequence_num: i32,
    /// The raw page text. CR (`\r`) is the paragraph delimiter; cleaned later.
    pub body: String,
    /// Footnote text block, `null` when the page has none.
    pub footnotes: Option<String>,
    /// Unknown shape in samples (always `null`).
    pub hints: Option<serde_json::Value>,
    /// Unknown shape in samples (always `null`).
    pub services_raw: Option<serde_json::Value>,
}

// ---------------------------------------------------------------------------
// 2. Book enrichment (rig Extractor output + JSON cache record)
// ---------------------------------------------------------------------------

/// Output of the per-book enrichment Extractor call. `JsonSchema` is derived so
/// rig generates the tool schema — never hand-written. Also serialized into
/// `cache/enrichments/<book_id>.json`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct BookEnrichment {
    pub madhhab: Option<Madhhab>,
    pub aqeedah_school: Option<AqeedahSchool>,
    pub sect: Sect,
    /// Canonicity tier `1..=5`.
    pub canonicity_tier: u8,
    pub is_critical_edition: bool,
    /// The model's own confidence `0.0..=1.0`. Drives the Haiku→Sonnet retry.
    pub confidence: f32,
    /// 1–2 sentence justification, kept for auditing.
    pub rationale: String,
}

impl BookEnrichment {
    /// Placeholder used when enrichment is skipped (`--skip-enrichment`):
    /// no madhhab/aqeedah, `unknown` sect, and a neutral in-range
    /// `canonicity_tier = 3`.
    pub fn skipped() -> Self {
        BookEnrichment {
            madhhab: None,
            aqeedah_school: None,
            sect: Sect::Unknown,
            canonicity_tier: 3,
            is_critical_edition: false,
            confidence: 0.0,
            rationale: "enrichment skipped".to_string(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum Madhhab {
    Hanafi,
    Maliki,
    Shafii,
    Hanbali,
}

impl Madhhab {
    pub fn as_str(self) -> &'static str {
        match self {
            Madhhab::Hanafi => "hanafi",
            Madhhab::Maliki => "maliki",
            Madhhab::Shafii => "shafii",
            Madhhab::Hanbali => "hanbali",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum AqeedahSchool {
    Athari,
    Ashari,
    Maturidi,
}

impl AqeedahSchool {
    pub fn as_str(self) -> &'static str {
        match self {
            AqeedahSchool::Athari => "athari",
            AqeedahSchool::Ashari => "ashari",
            AqeedahSchool::Maturidi => "maturidi",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum Sect {
    Sunni,
    Shii,
    Ibadi,
    Mutazili,
    Unknown,
}

impl Sect {
    pub fn as_str(self) -> &'static str {
        match self {
            Sect::Sunni => "sunni",
            Sect::Shii => "shii",
            Sect::Ibadi => "ibadi",
            Sect::Mutazili => "mutazili",
            Sect::Unknown => "unknown",
        }
    }
}

// ---------------------------------------------------------------------------
// 3. Derived classifications stored on the chunk
// ---------------------------------------------------------------------------

/// Author era, derived mechanically from `main_author_death_hijri`.
/// Stored on the chunk as its lowercase string ([`Era::as_str`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Era {
    Sahaba,
    Tabiin,
    Mutaqaddim,
    Mutaakhir,
    Late,
    Modern,
}

impl Era {
    /// Spec'd mapping: ≤100 sahaba, ≤200 tabiin, ≤400 mutaqaddim,
    /// ≤1000 mutaakhir, ≤1300 late, otherwise (incl. the 99999 sentinel) modern.
    pub fn from_death_hijri(death_hijri: i32) -> Era {
        match death_hijri {
            d if d <= 100 => Era::Sahaba,
            d if d <= 200 => Era::Tabiin,
            d if d <= 400 => Era::Mutaqaddim,
            d if d <= 1000 => Era::Mutaakhir,
            d if d <= 1300 => Era::Late,
            _ => Era::Modern,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Era::Sahaba => "sahaba",
            Era::Tabiin => "tabiin",
            Era::Mutaqaddim => "mutaqaddim",
            Era::Mutaakhir => "mutaakhir",
            Era::Late => "late",
            Era::Modern => "modern",
        }
    }
}

// ---------------------------------------------------------------------------
// 4. The LanceDB row contract + per-book result
// ---------------------------------------------------------------------------

/// One row written to the `chunks` LanceDB table. Fields are grouped: identity,
/// denormalized book metadata, enriched book metadata, mechanical chunk-level
/// signals, then the vector. `embedding` lands in Arrow as
/// `FixedSizeList<Float32, 1024>`.
#[derive(Debug, Clone, Serialize)]
pub struct Chunk {
    /// `"{book_id}:{page_id}:{chunk_idx}"`.
    pub chunk_id: String,
    pub book_id: i64,
    pub page_id: i64,
    pub page_num: i32,
    pub sequence_num: i32,
    pub chunk_idx: i32,
    /// Cleaned, chunked text.
    pub text: String,

    // denormalized book-level (from book_metadata.json)
    pub title_ar: String,
    pub main_author_name_ar: String,
    pub main_author_death_hijri: i32,
    pub category_id: i32,
    pub category_name_ar: String,

    // enriched book-level (from Claude); `None`/"unknown" when skipped
    pub madhhab: Option<String>,
    pub aqeedah_school: Option<String>,
    pub sect: String,
    pub era: String,
    pub canonicity_tier: i32,
    pub is_critical_edition: bool,

    // chunk-level mechanical
    pub chapter_path: Vec<String>,
    pub contains_quran: bool,
    /// Always `false` for now; backfilled later from `page_isnads.parquet`.
    pub contains_isnad: bool,

    // vector — 1024-dim BGE-M3
    pub embedding: Vec<f32>,
}

/// Per-book outcome reported by a successful per-book task.
#[derive(Debug, Clone, Serialize)]
pub struct BookSummary {
    pub book_id: i64,
    pub title_ar: String,
    /// Number of `Chunk` rows written to LanceDB.
    pub chunk_count: usize,
    /// Pages streamed from `pages.jsonl`.
    pub page_count: usize,
    /// `true` when enrichment used the cache rather than calling Claude.
    pub enrichment_cached: bool,
}
