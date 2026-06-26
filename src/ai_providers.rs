//! The process-wide Anthropic client.
//!
//! Like [`crate::http::http_client`], this is a single shared instance behind a
//! `Lazy` rather than a value threaded through every call site. It reuses the
//! shared `reqwest` client and reads `ANTHROPIC_API_KEY` from `.env` at compile
//! time (via `dotenv!`), so a missing key is a compile error, matching how
//! `main` already treats it.

use dotenv_codegen::dotenv;
use once_cell::sync::Lazy;
use rig::providers::anthropic::Client as AnthropicClient;
use rig::providers::openai::Client as OpenAiClient;

use crate::http::http_client;

static ANTHROPIC: Lazy<AnthropicClient> = Lazy::new(|| {
    AnthropicClient::builder()
        .http_client(http_client().clone())
        .api_key(dotenv!("ANTHROPIC_API_KEY"))
        .build()
        .expect("failed to build Anthropic client")
});

static CHATGPT: Lazy<OpenAiClient> = Lazy::new(|| {
    OpenAiClient::builder()
        .http_client(http_client().clone())
        .api_key(dotenv!("OPENAI_API_KEY"))
        .build()
        .expect("failed to build OpenAI client")
});

pub fn anthropic_client() -> &'static AnthropicClient {
    &ANTHROPIC
}

pub fn openai_client() -> &'static OpenAiClient {
    &CHATGPT
}
