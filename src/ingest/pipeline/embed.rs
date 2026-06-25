//! Stage 6 — turn chunk texts into 1024-dim BGE-M3 vectors.
//!
//! Two backends behind [`Embedder`]:
//!   * `Mock` — deterministic pseudo-random unit vectors (seeded by the text),
//!     so the whole pipeline runs before the GPU service exists.
//!   * `Http` — `POST {BGE_M3_URL}` with the minimal contract
//!     `{"texts": [...]}` → `{"embeddings": [[...; 1024], ...]}`.
//!
//! Awaiting `embed` inside the per-book loop is what provides backpressure: a
//! book cannot race ahead of the embedder because it blocks here before reading
//! more pages.

use crate::ingest::error::EmbedError;
use dotenv_codegen::dotenv;
use regex::Regex;
use rig::embeddings::{Embedding, EmbeddingError, EmbeddingModel};
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::OnceLock;

/// BGE-M3 output dimensionality.
pub const EMBED_DIM: usize = 1024;

/// Normalize text **for embedding only** — never for the stored `text` column,
/// which keeps the original vowelled, span-tagged form for display/rendering.
///
/// Three transforms:
///   * strip `<span …>` / `</span>` heading-marker tags (keeping their inner
///     text, which is real topical content like «فصل في القصر والجمع»);
///   * remove Arabic tashkil — harakat, tanwin, superscript alef, the Quranic
///     annotation marks — and the tatweel elongation `ـ`;
///   * fold orthographic letter variants ([`fold_orthography`]): hamzated alefs
///     → bare alef, teh marbuta → heh, alef maqsura → yeh.
///
/// BGE-M3's XLM-RoBERTa tokenizer was trained mostly on *unvowelled* Arabic, so
/// a heavily vowelled corpus tokenizes differently from a bare query of the same
/// words. Normalizing collapses that mismatch — and the letter folding collapses
/// the spelling inconsistencies (missing hamzas, ة/ه, ى/ي) that are pervasive in
/// classical texts and would otherwise break lexical matching. It is applied
/// symmetrically at the embedding boundary (documents during ingest, queries at
/// search time) so the index and the query share one canonical surface form.
pub fn normalize_for_embedding(text: &str) -> String {
    let no_span = strip_span_tags(text);
    let no_tashkil = span_tashkil_re().replace_all(&no_span, "");
    fold_orthography(&no_tashkil)
}

/// Fold Arabic orthographic variants to one canonical letter so that spelling
/// inconsistencies common in classical texts don't fragment lexical matching:
///   * أ (U+0623), إ (U+0625), آ (U+0622) → ا (U+0627), bare alef;
///   * ة (U+0629) → ه (U+0647), teh marbuta to heh;
///   * ى (U+0649) → ي (U+064A), alef maqsura to yeh.
///
/// Applied to documents and queries alike via [`normalize_for_embedding`], so
/// both sides of a search match on the same folded form.
fn fold_orthography(text: &str) -> String {
    text.chars()
        .map(|c| match c {
            'أ' | 'إ' | 'آ' => 'ا',
            'ة' => 'ه',
            'ى' => 'ي',
            other => other,
        })
        .collect()
}

/// Strip `<span …>`/`</span>` heading-marker tags, keeping their inner text.
/// Used at display time (the stored `text` column keeps the tags because the
/// ingest chunker splits sections on them) — and by [`normalize_for_embedding`].
pub fn strip_span_tags(text: &str) -> String {
    span_tag_re().replace_all(text, "").into_owned()
}

/// `<span …>` or `</span>` opener/closer, any attributes, case-insensitive.
fn span_tag_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?is)</?span[^>]*>").unwrap())
}

/// Arabic combining diacritics + tatweel. Ranges: signs (U+0610–061A),
/// harakat/tanwin (U+064B–065F), superscript alef (U+0670), Quranic marks
/// (U+06D6–06DC, U+06DF–06E8, U+06EA–06ED), tatweel (U+0640).
fn span_tashkil_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"[\u{0610}-\u{061A}\u{064B}-\u{065F}\u{0670}\u{06D6}-\u{06DC}\u{06DF}-\u{06E8}\u{06EA}-\u{06ED}\u{0640}]",
        )
        .unwrap()
    })
}

/// Embedding backend.
pub enum Embedder {
    Mock,
    Http {
        url: String,
        client: reqwest::Client,
    },
}

#[derive(Serialize)]
struct EmbedRequest<'a> {
    texts: &'a [String],
}

#[derive(Deserialize)]
struct EmbedResponse {
    embeddings: Vec<Vec<f32>>,
}

impl Embedder {
    /// Build the backend from the CLI/env: `Mock` when `--mock-embeddings`,
    /// otherwise an HTTP client pointed at `$BGE_M3_URL`.
    pub fn from_config(mock: bool) -> Result<Embedder, EmbedError> {
        if mock {
            return Ok(Embedder::Mock);
        }
        let url = dotenv!("BGE_M3_URL");
        Ok(Embedder::Http {
            url: String::from(url),
            client: reqwest::Client::new(),
        })
    }

    /// Embed a batch of texts; returns one 1024-dim vector per input, in order.
    /// Inputs are run through [`normalize_for_embedding`] first so the document
    /// vectors match queries (which are normalized the same way); the caller's
    /// original text is untouched and is what gets stored.
    pub async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbedError> {
        let texts: Vec<String> = texts.iter().map(|t| normalize_for_embedding(t)).collect();
        let texts = texts.as_slice();
        match self {
            Embedder::Mock => Ok(texts.iter().map(|t| mock_vector(t)).collect()),
            Embedder::Http { url, client } => {
                let resp = client
                    .post(url)
                    .json(&EmbedRequest { texts })
                    .send()
                    .await?
                    .error_for_status()?
                    .json::<EmbedResponse>()
                    .await?;

                if resp.embeddings.len() != texts.len() {
                    return Err(EmbedError::CountMismatch {
                        sent: texts.len(),
                        got: resp.embeddings.len(),
                    });
                }
                if let Some(bad) = resp.embeddings.iter().find(|v| v.len() != EMBED_DIM) {
                    return Err(EmbedError::DimMismatch {
                        expected: EMBED_DIM,
                        got: bad.len(),
                    });
                }
                Ok(resp.embeddings)
            }
        }
    }
}

/// Deterministic unit vector for a text: xorshift64 seeded by the text hash,
/// mapped to `[-1, 1)`, then L2-normalized.
fn mock_vector(text: &str) -> Vec<f32> {
    let mut hasher = DefaultHasher::new();
    text.hash(&mut hasher);
    let mut state = hasher.finish() | 1; // never zero

    let mut v = Vec::with_capacity(EMBED_DIM);
    for _ in 0..EMBED_DIM {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let unit = (state >> 11) as f64 / (1u64 << 53) as f64; // [0, 1)
        v.push((unit * 2.0 - 1.0) as f32);
    }

    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

#[derive(Clone)]
#[allow(dead_code)]
pub struct LocalBgeM3Model {
    client: reqwest::Client,
    url: String,
}

impl EmbeddingModel for LocalBgeM3Model {
    const MAX_DOCUMENTS: usize = 0;
    type Client = reqwest::Client;

    fn make(client: &Self::Client, _model: impl Into<String>, dims: Option<usize>) -> Self {
        // Match or default the dimensions tracking. BGE-M3 expects exactly 1024.
        let verified_dims = dims.unwrap_or(1024);
        if verified_dims != 1024 {
            eprintln!(
                "Warning: BGE-M3 natively generates 1024-dimensional vectors; received request for {}",
                verified_dims
            );
        }

        // Return your custom Rig _model instance mapped out to hit your pipeline script endpoint
        // You can hardcode your orchestration URL or use a default path format hook
        LocalBgeM3Model {
            client: client.clone(),
            url: dotenv!("BGE_M3_URL").to_string(),
        }
    }

    // BGE-M3's dense width configured in your script
    fn ndims(&self) -> usize {
        1024
    }

    async fn embed_texts(
        &self,
        documents: impl IntoIterator<Item = String> + Send,
    ) -> Result<Vec<Embedding>, EmbeddingError> {
        let texts: Vec<String> = documents.into_iter().collect();
        if texts.is_empty() {
            return Ok(vec![]);
        }

        // Normalize the same way ingest does, so a query embeds identically to
        // how the matching document was embedded. `texts` (the originals) stays
        // intact for the `document` field below.
        let norm_texts: Vec<String> =
            texts.iter().map(|t| normalize_for_embedding(t)).collect();
        let payload = EmbedRequest { texts: &norm_texts };

        // Hit your FastAPI /embed endpoint
        let response: EmbedResponse = self
            .client
            .post(&self.url)
            .json(&payload)
            .send()
            .await
            .map_err(|e| EmbeddingError::ProviderError(e.to_string()))?
            .json()
            .await
            .map_err(|e| EmbeddingError::ProviderError(e.to_string()))?;

        // Rig expects an iterator of `Embedding` structs containing the raw text and its f64/f32 vector projection
        let embeddings = texts
            .into_iter()
            .zip(response.embeddings.into_iter())
            .map(|(text, vec)| Embedding {
                document: text,
                // Rig vectors can use f64 arrays or matched f32 vectors depending on rig core version
                vec: vec.into_iter().map(|v| v as f64).collect(),
            })
            .collect();

        Ok(embeddings)
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn mock_is_deterministic_unit_dim() {
        let e = Embedder::Mock;
        let a = e.embed(&["نص".to_string()]).await.unwrap();
        let b = e.embed(&["نص".to_string()]).await.unwrap();
        assert_eq!(a.len(), 1);
        assert_eq!(a[0].len(), EMBED_DIM);
        assert_eq!(a, b, "same text must give the same vector");

        let norm = a[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4, "vector must be unit length");
    }

    #[tokio::test]
    async fn mock_differs_by_text() {
        let e = Embedder::Mock;
        let a = e.embed(&["alpha".to_string()]).await.unwrap();
        let b = e.embed(&["beta".to_string()]).await.unwrap();
        assert_ne!(a[0], b[0]);
    }

    #[test]
    fn normalize_strips_span_tags_keeping_inner_text() {
        let input = "<span data-type=\"title\" id=toc-30>فصل في القصر والجمع</span> ونص";
        assert_eq!(normalize_for_embedding(input), "فصل في القصر والجمع ونص");
    }

    #[test]
    fn normalize_strips_tashkil_and_tatweel() {
        // وَمُسَافِرٍ + tatweel -> bare letters.
        assert_eq!(normalize_for_embedding("وَمُسَافِرٍ يَقْصُرُ"), "ومسافر يقصر");
        assert_eq!(normalize_for_embedding("الْـــحَمْدُ"), "الحمد");
    }

    #[test]
    fn normalize_preserves_plain_arabic_and_quran_brackets() {
        // No diacritics, no spans, no foldable letters: unchanged, including the
        // ﴿﴾ Quran ornaments.
        let s = "قصر الصلاة ﴿نص﴾ للمسافر";
        assert_eq!(normalize_for_embedding(s), s);
    }

    #[test]
    fn normalize_folds_orthographic_variants() {
        // Hamzated alefs collapse to bare alef, ة → ه, ى → ي.
        assert_eq!(normalize_for_embedding("أإآ"), "ااا");
        assert_eq!(normalize_for_embedding("صلاة"), "صلاه");
        assert_eq!(normalize_for_embedding("مصطفى"), "مصطفي");
        // أية (with hamza) and اية (without) normalize to the same form, so a
        // query spelled either way matches the corpus.
        assert_eq!(normalize_for_embedding("أية"), normalize_for_embedding("اية"));
    }

    #[tokio::test]
    async fn mock_vector_is_invariant_under_normalization() {
        // Same word vowelled vs. bare must embed identically once normalized.
        let e = Embedder::Mock;
        let vowelled = e.embed(&["مُسَافِر".to_string()]).await.unwrap();
        let bare = e.embed(&["مسافر".to_string()]).await.unwrap();
        assert_eq!(vowelled[0], bare[0]);
    }
}
