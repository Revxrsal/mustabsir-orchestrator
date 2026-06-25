//! `verify` — confirm the Rust BGE-M3 port behaves, and (optionally) that it
//! matches the vectors already in LanceDB. Rust analogue of `../bge-m3/verify.py`.
//!
//! Two modes:
//!
//!   cargo run --release --bin verify
//!       Self-tests, no external data: dim==1024, unit norm, determinism, and a
//!       semantic sanity check (a related Arabic pair must out-score an unrelated
//!       one). Exits non-zero on failure.
//!
//!   cargo run --release --bin verify -- ref.json
//!       Real compatibility check. `ref.json` is one row exported from the
//!       LanceDB the Python service built:
//!           {"text": "<the embedded surface form>", "vector": [ .. 1024 .. ]}
//!       The tool embeds `text` with this engine and prints cosine vs the stored
//!       vector. Expect ~0.999 (fp16/ONNX drift); <0.99 means a real mismatch
//!       (pooling / max_length / quantization / normalization) — investigate.
//!
//! NOTE: pass `text` already in the form the index stored — i.e. spans + tashkil
//! stripped, the way `embed.rs::normalize_for_embedding` produces. The service
//! embeds verbatim; this tool does too, so what you feed must already be normalized.

use std::process::ExitCode;

use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};
use serde::Deserialize;

const EMBED_DIM: usize = 1024;
const MAX_LENGTH: usize = 2048;

#[derive(Deserialize)]
struct Reference {
    text: String,
    vector: Vec<f32>,
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 { 0.0 } else { dot / (na * nb) }
}

fn l2(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn build_model() -> anyhow::Result<TextEmbedding> {
    // Same settings as the service — anything else would make verify lie.
    Ok(TextEmbedding::try_new(
        TextInitOptions::new(EmbeddingModel::BGEM3)
            .with_max_length(MAX_LENGTH)
            .with_show_download_progress(true),
    )?)
}

fn embed_one(model: &mut TextEmbedding, text: &str) -> anyhow::Result<Vec<f32>> {
    let mut out = model.embed(vec![text.to_string()], None)?;
    Ok(out.remove(0))
}

fn main() -> ExitCode {
    match run() {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> anyhow::Result<bool> {
    println!("loading BAAI/bge-m3 (fp32 ONNX, CPU, max_length={MAX_LENGTH}) ...");
    let mut model = build_model()?;
    println!("model ready\n");

    if let Some(path) = std::env::args().nth(1) {
        return reference_check(&mut model, &path);
    }
    self_tests(&mut model)
}

/// Compatibility check against a LanceDB-exported row.
fn reference_check(model: &mut TextEmbedding, path: &str) -> anyhow::Result<bool> {
    let raw = std::fs::read_to_string(path)?;
    let reference: Reference = serde_json::from_str(&raw)?;

    if reference.vector.len() != EMBED_DIM {
        anyhow::bail!(
            "reference vector is {}-dim, expected {EMBED_DIM}",
            reference.vector.len()
        );
    }

    let got = embed_one(model, &reference.text)?;
    let cos = cosine(&got, &reference.vector);

    println!("text          : {}", reference.text);
    println!("stored ‖v‖    : {:.6}", l2(&reference.vector));
    println!("rust   ‖v‖    : {:.6}", l2(&got));
    println!("cosine(rust, stored) = {cos:.6}");

    // ~0.999 is the expected fp16/ONNX drift; allow margin but flag real breaks.
    let pass = cos >= 0.99;
    println!(
        "\n{}",
        if pass {
            "PASS — vectors match the index (drift within fp16/ONNX tolerance)"
        } else {
            "FAIL — cosine too low; this is a method mismatch, not drift. \
             Check pooling / max_length / quantization / pre-normalization."
        }
    );
    Ok(pass)
}

/// Self-contained invariants + semantic sanity, no external data.
fn self_tests(model: &mut TextEmbedding) -> anyhow::Result<bool> {
    let mut ok = true;
    let mut check = |name: &str, cond: bool| {
        println!("[{}] {name}", if cond { "PASS" } else { "FAIL" });
        ok &= cond;
    };

    // Inputs are pre-normalized (no tashkil/spans), as the index stores them.
    let a = embed_one(model, "قصر الصلاة للمسافر")?; // shorten prayer for traveler
    let b = embed_one(model, "جمع الصلاة في السفر")?; // combine prayer while traveling (related)
    let c = embed_one(model, "احكام زكاة الذهب والفضة")?; // zakat on gold/silver (unrelated)
    let a2 = embed_one(model, "قصر الصلاة للمسافر")?; // same as `a`

    check("dim == 1024", a.len() == EMBED_DIM);
    check("unit norm (‖a‖ ≈ 1)", (l2(&a) - 1.0).abs() < 1e-3);
    check("deterministic (a == a2)", a == a2);

    let related = cosine(&a, &b);
    let unrelated = cosine(&a, &c);
    println!("    cosine(related)   = {related:.4}");
    println!("    cosine(unrelated) = {unrelated:.4}");
    check("related > unrelated", related > unrelated);

    println!(
        "\n{}",
        if ok { "ALL PASS" } else { "FAILURES — see above" }
    );
    Ok(ok)
}
