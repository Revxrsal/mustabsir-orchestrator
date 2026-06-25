//! Stage 5 — book-level enrichment via a rig `Extractor`, with a JSON cache.
//!
//! One Extractor call per book produces a [`BookEnrichment`]. We try Haiku
//! first; if the model's own `confidence` is below [`CONFIDENCE_THRESHOLD`] we
//! retry once with Sonnet and keep that result. The final result is cached at
//! `<cache_root>/<book_id>.json`; re-running enrichment is a manual `rm`.
//!
//! The system prompt is versioned as `prompts/book_enrichment.md` and embedded
//! at compile time — never an inline string.

use std::path::Path;

use rig::client::CompletionClient;
use rig::providers::anthropic::Client as AnthropicClient;
use rig::providers::anthropic::completion::{CLAUDE_HAIKU_4_5, CLAUDE_SONNET_4_6};

use crate::ingest::error::EnrichError;
use crate::ingest::model::{BookEnrichment, BookMetadata, DEATH_HIJRI_UNKNOWN};
use crate::ingest::pipeline::pages;

const SYSTEM_PROMPT: &str = include_str!("../../../prompts/book_enrichment.md");
/// Below this self-reported confidence, retry once with the stronger model.
const CONFIDENCE_THRESHOLD: f32 = 0.7;
/// Characters of the first page body fed into the prompt.
const MAX_EXCERPT_CHARS: usize = 2000;
/// Output token budget for the extractor (the JSON is small).
const MAX_TOKENS: u64 = 1024;

/// Enrich one book. Returns `(enrichment, from_cache)`.
#[tracing::instrument(skip_all, fields(book_id = meta.book_id))]
pub async fn enrich(
    meta: &BookMetadata,
    first_page_body: Option<&str>,
    client: &AnthropicClient,
    cache_root: &Path,
) -> Result<(BookEnrichment, bool), EnrichError> {
    let cache_path = cache_root.join(format!("{}.json", meta.book_id));
    if let Some(cached) = read_cache(&cache_path)? {
        tracing::debug!("enrichment cache hit");
        return Ok((cached, true));
    }

    let input = build_input(meta, first_page_body);

    // Haiku first.
    let mut result = extract_with(client, CLAUDE_HAIKU_4_5, &input).await?;
    let mut model_used = "haiku";

    // Escalate once to Sonnet on low confidence.
    if result.confidence < CONFIDENCE_THRESHOLD {
        tracing::info!(
            confidence = result.confidence,
            "low confidence; retrying with Sonnet"
        );
        result = extract_with(client, CLAUDE_SONNET_4_6, &input).await?;
        model_used = "sonnet";
    }

    tracing::info!(
        model = model_used,
        confidence = result.confidence,
        sect = result.sect.as_str(),
        tier = result.canonicity_tier,
        "enriched book",
    );

    write_cache(&cache_path, &result)?;
    Ok((result, false))
}

async fn extract_with(
    client: &AnthropicClient,
    model: &str,
    input: &str,
) -> Result<BookEnrichment, EnrichError> {
    client
        .extractor::<BookEnrichment>(model)
        .preamble(SYSTEM_PROMPT)
        .max_tokens(MAX_TOKENS)
        .build()
        .extract(input.to_string())
        .await
        .map_err(|e| EnrichError::Llm(e.to_string()))
}

/// Assemble the prompt input from the metadata and the opening excerpt.
fn build_input(meta: &BookMetadata, first_page_body: Option<&str>) -> String {
    let death = if meta.main_author_death_hijri == DEATH_HIJRI_UNKNOWN {
        "غير معروف (معاصر أو لم تُذكر)".to_string()
    } else {
        format!("{} هـ", meta.main_author_death_hijri)
    };

    let excerpt = first_page_body
        .map(pages::clean)
        .map(|c| c.chars().take(MAX_EXCERPT_CHARS).collect::<String>())
        .unwrap_or_default();

    format!(
        "العنوان: {title}\n\
         المؤلف: {author}\n\
         وفاة المؤلف: {death}\n\
         تصنيف الشاملة: {category}\n\
         \n\
         بطاقة الكتاب:\n{betaka}\n\
         \n\
         مقتطف من بداية الكتاب:\n{excerpt}",
        title = meta.title_ar,
        author = meta.main_author_name_ar,
        category = meta.category_name_ar,
        betaka = meta.betaka_text,
    )
}

fn read_cache(path: &Path) -> Result<Option<BookEnrichment>, EnrichError> {
    match std::fs::read_to_string(path) {
        Ok(s) => Ok(Some(serde_json::from_str(&s)?)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(EnrichError::Cache(e)),
    }
}

fn write_cache(path: &Path, enrichment: &BookEnrichment) -> Result<(), EnrichError> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, serde_json::to_string_pretty(enrichment)?)?;
    Ok(())
}
