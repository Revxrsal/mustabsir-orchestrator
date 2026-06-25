//! Stage 5 — TOC-aware chunking.
//!
//! Default strategy (do not change without asking): split a cleaned page body on
//! `<span data-type='title'>` heading boundaries; text before the first heading
//! is its own section. Consecutive sections are then **packed** into groups of at
//! least `TARGET_TOKENS` whitespace tokens (so we don't emit a flood of tiny
//! heading-only fragments), and each group is finally sliced with a
//! 600-whitespace-token window / 80-token overlap. Token count =
//! `split_whitespace().count()` — no tokenizer dependency.
//!
//! Packing is bounded to a single page: `chunk_page` runs per page, so a group
//! never crosses a page break (that boundary is still handled upstream by the
//! page-straddling tail carry-forward, [`tail_tokens`]).
//!
//! `chunk_idx` increments across the whole page. Chunks are returned with an
//! empty `embedding`; the embed stage fills it before the write stage.

use std::sync::OnceLock;

use regex::Regex;

use crate::ingest::model::{BookEnrichment, BookMetadata, Chunk, Era, PageRow};
use crate::ingest::pipeline::pages;

/// Whitespace-token window cap.
const CHUNK_CAP: usize = 600;
/// Whitespace-token overlap between adjacent windows.
const OVERLAP: usize = 80;
/// Target lower bound (in whitespace tokens) for a packed group of consecutive
/// sections: keep merging sections until the running total reaches this, then
/// flush. Stops short heading-bounded sections from each becoming their own
/// undersized chunk.
const TARGET_TOKENS: usize = 400;
/// Floor for a *trailing* group. If the final remainder after packing is smaller
/// than this, it is absorbed into the previous group instead of being emitted as
/// a tiny fragment. (A page whose entire body is below the floor still yields one
/// short chunk — there is nothing to merge it with.)
const MIN_TOKENS: usize = 120;

/// Chunk one cleaned page into `Chunk` rows (embeddings left empty).
pub fn chunk_page(
    cleaned: &str,
    page: &PageRow,
    meta: &BookMetadata,
    enrichment: &BookEnrichment,
    era: Era,
    chapter_path: &[String],
) -> Vec<Chunk> {
    let mut chunks = Vec::new();
    let mut idx: i32 = 0;

    let sections = split_on_headings(cleaned);
    for group in group_sections(&sections) {
        for text in window_tokens(&group) {
            chunks.push(make_chunk(idx, text, page, meta, enrichment, era, chapter_path));
            idx += 1;
        }
    }

    chunks
}

/// Greedily pack consecutive sections into groups of at least `TARGET_TOKENS`
/// whitespace tokens. Sections are contiguous slices of the page body, so a group
/// is just their concatenation. The final group may fall short of the target; if
/// it also falls below `MIN_TOKENS` it is folded into the previous group rather
/// than emitted on its own (unless it is the only group).
fn group_sections(sections: &[&str]) -> Vec<String> {
    let mut groups: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut cur_tokens = 0usize;

    for &section in sections {
        if !cur.is_empty() {
            cur.push(' ');
        }
        cur.push_str(section);
        cur_tokens += section.split_whitespace().count();

        if cur_tokens >= TARGET_TOKENS {
            groups.push(std::mem::take(&mut cur));
            cur_tokens = 0;
        }
    }

    // Flush the trailing remainder, absorbing it into the previous group if it is
    // an undersized fragment.
    if !cur.trim().is_empty() {
        if cur_tokens < MIN_TOKENS && !groups.is_empty() {
            let last = groups.last_mut().unwrap();
            last.push(' ');
            last.push_str(&cur);
        } else {
            groups.push(cur);
        }
    }

    groups
}

/// Split text into sections at heading-span starts. The slice before the first
/// heading (if any non-whitespace) is the first section.
fn split_on_headings(text: &str) -> Vec<&str> {
    let starts: Vec<usize> = heading_re().find_iter(text).map(|m| m.start()).collect();

    if starts.is_empty() {
        return if text.trim().is_empty() {
            vec![]
        } else {
            vec![text]
        };
    }

    let mut sections = Vec::new();
    // Text before the first heading.
    if starts[0] > 0 && !text[..starts[0]].trim().is_empty() {
        sections.push(&text[..starts[0]]);
    }
    // Each heading start to the next heading start.
    for (i, &start) in starts.iter().enumerate() {
        let end = starts.get(i + 1).copied().unwrap_or(text.len());
        let section = &text[start..end];
        if !section.trim().is_empty() {
            sections.push(section);
        }
    }
    sections
}

/// The last `n` whitespace tokens of `text`, rejoined with single spaces (empty
/// if `text` is blank). Used to carry a tail of one page onto the front of the
/// next so a passage straddling a page break is captured whole on each side.
pub(crate) fn tail_tokens(text: &str, n: usize) -> String {
    let tokens: Vec<&str> = text.split_whitespace().collect();
    let start = tokens.len().saturating_sub(n);
    tokens[start..].join(" ")
}

/// Slide a `CHUNK_CAP`-token window with `OVERLAP` overlap over a section,
/// rejoining tokens with single spaces.
fn window_tokens(section: &str) -> Vec<String> {
    let tokens: Vec<&str> = section.split_whitespace().collect();
    if tokens.is_empty() {
        return vec![];
    }

    let step = CHUNK_CAP - OVERLAP;
    let mut out = Vec::new();
    let mut start = 0;
    loop {
        let end = (start + CHUNK_CAP).min(tokens.len());
        out.push(tokens[start..end].join(" "));
        if end == tokens.len() {
            break;
        }
        start += step;
    }
    out
}

fn make_chunk(
    idx: i32,
    text: String,
    page: &PageRow,
    meta: &BookMetadata,
    enrichment: &BookEnrichment,
    era: Era,
    chapter_path: &[String],
) -> Chunk {
    let contains_quran = pages::contains_quran(&text);
    Chunk {
        chunk_id: format!("{}:{}:{}", page.book_id, page.page_id, idx),
        book_id: page.book_id,
        page_id: page.page_id,
        // -1 sentinel for pages with no printed page number (source `null`).
        // Keeps `Chunk.page_num`/the LanceDB column non-nullable Int32.
        page_num: page.page_num.unwrap_or(-1),
        sequence_num: page.sequence_num,
        chunk_idx: idx,
        text,

        title_ar: meta.title_ar.clone(),
        main_author_name_ar: meta.main_author_name_ar.clone(),
        main_author_death_hijri: meta.main_author_death_hijri,
        category_id: meta.category_id,
        category_name_ar: meta.category_name_ar.clone(),

        madhhab: enrichment.madhhab.map(|m| m.as_str().to_string()),
        aqeedah_school: enrichment.aqeedah_school.map(|a| a.as_str().to_string()),
        sect: enrichment.sect.as_str().to_string(),
        era: era.as_str().to_string(),
        canonicity_tier: enrichment.canonicity_tier as i32,
        is_critical_edition: enrichment.is_critical_edition,

        chapter_path: chapter_path.to_vec(),
        contains_quran,
        contains_isnad: false,

        embedding: Vec::new(),
    }
}

// Heading span opener. Lenient about attribute order and quoting: the samples
// use `<span data-type="title" id=toc-1>` (note the unquoted `id`).
fn heading_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r#"(?s)<span[^>]*data-type\s*=\s*["']?title["']?[^>]*>"#).unwrap()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SPAN: &str = "<span data-type=\"title\" id=toc-1>عنوان</span>";

    #[test]
    fn no_heading_is_single_section() {
        let s = split_on_headings("just some body text");
        assert_eq!(s, vec!["just some body text"]);
    }

    #[test]
    fn empty_text_yields_no_sections() {
        assert!(split_on_headings("   \n  ").is_empty());
    }

    #[test]
    fn text_before_first_heading_is_own_section() {
        let text = format!("intro words {SPAN} body one {SPAN} body two");
        let sections = split_on_headings(&text);
        assert_eq!(sections.len(), 3);
        assert_eq!(sections[0], "intro words ");
        assert!(sections[1].starts_with("<span"));
        assert!(sections[1].contains("body one"));
        assert!(sections[2].contains("body two"));
    }

    #[test]
    fn heading_at_start_has_no_leading_section() {
        let text = format!("{SPAN} body one");
        let sections = split_on_headings(&text);
        assert_eq!(sections.len(), 1);
        assert!(sections[0].starts_with("<span"));
    }

    #[test]
    fn tail_tokens_takes_last_n_and_clamps() {
        assert_eq!(tail_tokens("a b c d e", 2), "d e");
        assert_eq!(tail_tokens("a b", 5), "a b"); // fewer than n -> all
        assert_eq!(tail_tokens("   ", 3), ""); // blank -> empty
        assert_eq!(tail_tokens("a b c", 0), ""); // n=0 -> empty
    }

    /// Build a section of `n` placeholder tokens (`"0 1 2 ..."`).
    fn section_of(n: usize) -> String {
        (0..n).map(|i| i.to_string()).collect::<Vec<_>>().join(" ")
    }

    #[test]
    fn group_sections_packs_small_sections_to_target() {
        // Three 150-token sections: 150 < 400, 300 < 400, 450 >= 400 -> one group.
        let secs = [section_of(150), section_of(150), section_of(150)];
        let refs: Vec<&str> = secs.iter().map(|s| s.as_str()).collect();
        let groups = group_sections(&refs);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].split_whitespace().count(), 450);
    }

    #[test]
    fn group_sections_flushes_each_time_target_is_reached() {
        // Four 250-token sections: flush at 500 (secs 0+1), flush at 500 (secs 2+3).
        let secs: Vec<String> = (0..4).map(|_| section_of(250)).collect();
        let refs: Vec<&str> = secs.iter().map(|s| s.as_str()).collect();
        let groups = group_sections(&refs);
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].split_whitespace().count(), 500);
        assert_eq!(groups[1].split_whitespace().count(), 500);
    }

    #[test]
    fn group_sections_absorbs_tiny_trailing_remainder() {
        // 450 -> flush; trailing 50 (< MIN_TOKENS) folds back into the first group.
        let secs = [section_of(450), section_of(50)];
        let refs: Vec<&str> = secs.iter().map(|s| s.as_str()).collect();
        let groups = group_sections(&refs);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].split_whitespace().count(), 500);
    }

    #[test]
    fn group_sections_keeps_a_substantial_remainder() {
        // 450 -> flush; trailing 200 (>= MIN_TOKENS) stands on its own.
        let secs = [section_of(450), section_of(200)];
        let refs: Vec<&str> = secs.iter().map(|s| s.as_str()).collect();
        let groups = group_sections(&refs);
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[1].split_whitespace().count(), 200);
    }

    #[test]
    fn group_sections_emits_lone_short_page_as_one_group() {
        // A whole page below the floor has nothing to merge with -> one short group.
        let secs = [section_of(30)];
        let refs: Vec<&str> = secs.iter().map(|s| s.as_str()).collect();
        let groups = group_sections(&refs);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].split_whitespace().count(), 30);
    }

    #[test]
    fn group_sections_empty_input_is_empty() {
        assert!(group_sections(&[]).is_empty());
    }

    #[test]
    fn short_section_is_one_window() {
        let section = "a b c d e";
        let w = window_tokens(section);
        assert_eq!(w, vec!["a b c d e".to_string()]);
    }

    #[test]
    fn windows_respect_cap_and_overlap() {
        // 700 tokens -> [0,600) and [520,700): two windows, 80 shared.
        let section = (0..700)
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join(" ");
        let windows = window_tokens(&section);
        assert_eq!(windows.len(), 2);

        let w0: Vec<&str> = windows[0].split_whitespace().collect();
        let w1: Vec<&str> = windows[1].split_whitespace().collect();
        assert_eq!(w0.len(), 600);
        assert_eq!(w1.len(), 180);
        // Overlap: last 80 of w0 == first 80 of w1, and the second window
        // starts exactly at token index 520 (= cap - overlap).
        assert_eq!(&w0[520..600], &w1[0..80]);
        assert_eq!(w1[0], "520");
    }

    #[test]
    fn exact_cap_is_single_window() {
        let section = (0..600)
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join(" ");
        assert_eq!(window_tokens(&section).len(), 1);
    }

    #[test]
    fn three_windows_chain() {
        // 1300 tokens: starts 0, 520, 1040 -> 3 windows (600, 600, 260).
        let section = (0..1300)
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join(" ");
        let windows = window_tokens(&section);
        assert_eq!(windows.len(), 3);
        let last: Vec<&str> = windows[2].split_whitespace().collect();
        assert_eq!(last.len(), 260);
        assert_eq!(last[0], "1040");
    }

    fn sample_page(body: &str) -> PageRow {
        PageRow {
            page_id: 42,
            book_id: 7,
            shamela_page_id: 1,
            part: None,
            page_num: Some(100),
            sequence_num: 3,
            body: body.to_string(),
            footnotes: None,
            hints: None,
            services_raw: None,
        }
    }

    fn sample_meta() -> BookMetadata {
        BookMetadata {
            book_id: 7,
            shamela_id: 7,
            title_ar: "كتاب".into(),
            book_type: 1,
            book_type_label: "كتاب".into(),
            category_id: 1,
            category_name_ar: "العقيدة".into(),
            main_author_id: 1,
            main_author_name_ar: "مؤلف".into(),
            main_author_death_hijri: 300,
            main_author_death_hijri_text: None,
            authors_text: "1".into(),
            authors: vec![],
            hijri_era: 300,
            printed: true,
            is_hidden: false,
            parent_id: None,
            group_id: None,
            version_major: 1,
            version_minor: 0,
            betaka_text: String::new(),
            meta: serde_json::Value::Null,
            volume_count_observed: 0,
            has_multi_part: false,
        }
    }

    #[test]
    fn chunk_page_sets_ids_paths_and_quran_flag() {
        let body = format!("{SPAN} فيه ﴿آية﴾ ونص بعدها");
        let page = sample_page(&body);
        let meta = sample_meta();
        let chapter = vec!["باب".to_string(), "فصل".to_string()];
        let chunks = chunk_page(
            &body,
            &page,
            &meta,
            &BookEnrichment::skipped(),
            Era::from_death_hijri(meta.main_author_death_hijri),
            &chapter,
        );
        assert_eq!(chunks.len(), 1);
        let c = &chunks[0];
        assert_eq!(c.chunk_id, "7:42:0");
        assert_eq!(c.chunk_idx, 0);
        assert_eq!(c.chapter_path, chapter);
        assert!(c.contains_quran);
        assert!(!c.contains_isnad);
        assert_eq!(c.era, "mutaqaddim");
        assert_eq!(c.sect, "unknown");
        assert!(c.embedding.is_empty());
    }
}
