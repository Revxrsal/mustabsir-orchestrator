//! Stage 4 — stream `pages.jsonl` one row at a time and clean each body.
//!
//! Cleaning rules, applied in order (see project spec):
//!   1. `\r` → `\n` (Shamela uses CR as the paragraph delimiter).
//!   2. Strip footnote markers `(¬N)` with ASCII or Arabic-Indic digits.
//!   3. Strip `<img ...>` tags entirely (including base64 data URIs).
//!   4. Preserve `﴿…﴾`, honorific codepoints, and `<span data-type='title'>` —
//!      i.e. they are simply never matched by the strippers above.
//! Rule 5 (`contains_quran`) is evaluated per chunk via [`contains_quran`].

use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::sync::OnceLock;

use regex::Regex;

use crate::ingest::error::PageError;
use crate::ingest::model::PageRow;

/// Stream the pages of a book lazily; each item is one parsed `PageRow`.
pub fn stream(
    book_dir: &Path,
) -> Result<impl Iterator<Item = Result<PageRow, PageError>>, PageError> {
    let path = book_dir.join("pages.jsonl");
    let reader = BufReader::new(File::open(path)?);
    let iter = serde_json::Deserializer::from_reader(reader)
        .into_iter::<PageRow>()
        .map(|r| r.map_err(PageError::from));
    Ok(iter)
}

/// Read just the first page of a book (for the enrichment excerpt).
pub fn first_page(book_dir: &Path) -> Result<Option<PageRow>, PageError> {
    stream(book_dir)?.next().transpose()
}

/// Apply the cleaning rules to a raw page body.
pub fn clean(body: &str) -> String {
    let normalized = body.replace('\r', "\n");
    let no_footnotes = footnote_re().replace_all(&normalized, "");
    let no_img = img_re().replace_all(&no_footnotes, "");
    no_img.into_owned()
}

/// Rule 5: at least one surviving `﴿…﴾` pair.
pub fn contains_quran(text: &str) -> bool {
    quran_re().is_match(text)
}

// `(¬N)` where N is one or more ASCII, Arabic-Indic, or Extended Arabic-Indic digits.
fn footnote_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\(¬[0-9\x{0660}-\x{0669}\x{06F0}-\x{06F9}]+\)").unwrap())
}

// Whole `<img ...>` tags. Base64 data URIs contain no `>`, so `[^>]*` is safe.
fn img_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?s)<img[^>]*>").unwrap())
}

fn quran_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"﴿[^﴾]*﴾").unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cr_becomes_lf() {
        assert_eq!(clean("a\rb\rc"), "a\nb\nc");
    }

    #[test]
    fn strips_ascii_and_arabic_footnote_markers() {
        assert_eq!(clean("متن(¬1) آخر(¬٢) و(¬١٣)"), "متن آخر و");
    }

    #[test]
    fn strips_img_tags_including_data_uri() {
        let input = r#"قبل <img src="data:image/png;base64,AAAA+/=="> بعد"#;
        assert_eq!(clean(input), "قبل  بعد");
    }

    #[test]
    fn preserves_quran_brackets_and_heading_spans() {
        let input = "<span data-type=\"title\" id=toc-1>عنوان</span>\r﴿إِنَّا﴾";
        let cleaned = clean(input);
        assert!(cleaned.contains("<span data-type=\"title\" id=toc-1>عنوان</span>"));
        assert!(cleaned.contains("﴿إِنَّا﴾"));
        assert!(contains_quran(&cleaned));
    }

    #[test]
    fn contains_quran_false_without_pair() {
        assert!(!contains_quran("نص عادي بدون قرآن"));
        assert!(!contains_quran("﴿ بلا غلق"));
    }
}
