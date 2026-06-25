//! `compare` — embed the same texts with this Rust engine AND the running Python
//! service, and report per-text cosine. This is the apples-to-apples check that
//! the port reproduces the vectors the index was built from.
//!
//!   cargo run --release --bin compare                 # uses $BGE_M3_URL or :8000
//!   cargo run --release --bin compare -- http://host:8000/embed
//!
//! Texts are pre-normalized (no tashkil/spans) and sent identically to both
//! sides — the client-side normalization isn't part of either engine, so leaving
//! it out keeps the comparison about the engines only. Expect cosine ~0.999
//! (the Python service runs fp16 on CUDA; this one runs fp32 on CPU).

use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};
use serde::{Deserialize, Serialize};

const EMBED_DIM: usize = 1024;
const MAX_LENGTH: usize = 2048;

/// Pre-normalized Arabic samples of varying length (last one is long enough to
/// exercise tokenization past short-chunk lengths).
const TEXTS: &[&str] = &[
    "قصر الصلاة للمسافر",
    "جمع الصلاة في السفر",
    "احكام زكاة الذهب والفضة",
    "صيام رمضان وقضاء الايام",
    "الطهارة شرط لصحة الصلاة ولا تصح صلاة بغير طهور",
    "اذا سافر المسلم مسافة القصر جاز له ان يقصر الصلاة الرباعية ركعتين وله ان يجمع بين الظهر والعصر وبين المغرب والعشاء جمع تقديم او تاخير على ما ورد في السنة",
];

#[derive(Serialize)]
struct EmbedRequest<'a> {
    texts: &'a [String],
}

#[derive(Deserialize)]
struct EmbedResponse {
    embeddings: Vec<Vec<f32>>,
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 { 0.0 } else { dot / (na * nb) }
}

fn main() -> anyhow::Result<()> {
    let url = std::env::args()
        .nth(1)
        .or_else(|| std::env::var("BGE_M3_URL").ok())
        .unwrap_or_else(|| "http://localhost:8000/embed".to_string());

    let texts: Vec<String> = TEXTS.iter().map(|s| s.to_string()).collect();

    // --- Python service (the reference the index was built with) ---
    println!("querying Python service at {url} ...");
    let py: EmbedResponse = reqwest::blocking::Client::new()
        .post(&url)
        .json(&EmbedRequest { texts: &texts })
        .send()?
        .error_for_status()?
        .json()?;
    anyhow::ensure!(
        py.embeddings.len() == texts.len(),
        "python returned {} vectors for {} texts",
        py.embeddings.len(),
        texts.len()
    );

    // --- Rust engine (fastembed, fp32 CPU) ---
    println!("loading BAAI/bge-m3 (fp32 ONNX, CPU, max_length={MAX_LENGTH}) ...");
    let mut model = TextEmbedding::try_new(
        TextInitOptions::new(EmbeddingModel::BGEM3)
            .with_max_length(MAX_LENGTH)
            .with_show_download_progress(true),
    )?;
    let rust = model.embed(texts.clone(), None)?;
    println!();

    // --- per-text cosine ---
    let mut min = f32::INFINITY;
    let mut sum = 0.0f32;
    println!("{:>6}  {:>8}  text", "idx", "cosine");
    for (i, (r, p)) in rust.iter().zip(py.embeddings.iter()).enumerate() {
        anyhow::ensure!(r.len() == EMBED_DIM && p.len() == EMBED_DIM, "bad dim at {i}");
        let c = cosine(r, p);
        min = min.min(c);
        sum += c;
        let preview: String = texts[i].chars().take(28).collect();
        println!("{i:>6}  {c:>8.5}  {preview}");
    }
    let mean = sum / rust.len() as f32;

    println!("\nmin cosine  = {min:.5}");
    println!("mean cosine = {mean:.5}");
    println!(
        "\n{}",
        if min >= 0.99 {
            "PASS — Rust port matches the Python/index vectors (within fp16/ONNX drift)"
        } else {
            "FAIL — a text diverged >0.01; that's a method mismatch, not drift. \
             Check pooling / max_length / quantization."
        }
    );
    Ok(())
}
