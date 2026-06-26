pub mod ai_providers;
pub mod arrow_cols;
pub mod context;
pub mod distill;
pub mod http;
pub mod ingest;
pub mod retrieval;
pub mod scope;
pub mod slide_tool;
pub mod usage;

use crate::ai_providers::anthropic_client;
use crate::context::expand_context;
use crate::distill::distill_query;
use crate::http::http_client;
use crate::ingest::pipeline::embed::{LocalBgeM3Model, strip_span_tags};
use crate::retrieval::hybrid_search;
use crate::scope::{extract_scope, scope_filter};
use crate::usage::{report_turns, report_usage};
use dotenv::dotenv;
use dotenv_codegen::dotenv;
use rig::agent::AgentBuilder;
use rig::client::CompletionClient;
use rig::completion::Prompt;
use rig::embeddings::EmbeddingModel;
use rig::providers::anthropic::completion::CLAUDE_SONNET_4_6;
use slide_tool::UpdateBuildContent;
use std::collections::HashSet;
use std::env;

const GENERATOR_PROMPT: &str = include_str!("../prompts/powerpoint_generator.md");

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenv().ok();
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    // The brief comes from CLI args; fall back to a sample topic. The default is
    // a Hanbali-fiqh query on «المسح على الخفين» as treated in «أخصر المختصرات»
    // and its glosses — a clean, well-bounded masʾala for testing.
    let brief = env::args().skip(1).collect::<Vec<_>>().join(" ");
    let brief = if brief.trim().is_empty() {
        "ما هو شرح حديث (فأحرق على من لا يخرج إلى الصلاة) في فتح الباري؟".to_string()
    } else {
        brief
    };

    let ppt_gen_dir = dotenv!("PPT_GEN_DIR");

    // Enable Anthropic prompt caching on the model. `with_prompt_caching` marks
    // the (large, static) system prompt and the final tool definition with
    // `cache_control` breakpoints, so they are billed at the ~10% cache-read rate
    // on every request after the first within the 5-minute TTL — both across the
    // turns of a single run and across back-to-back runs. The brief (the only
    // part that changes) stays uncached.
    let generation_model = anthropic_client()
        .completion_model(CLAUDE_SONNET_4_6)
        .with_prompt_caching();

    let embeddings_model = LocalBgeM3Model::make(http_client(), "bge-m3", None);

    let conn = lancedb::connect("lancedb").execute().await?;
    let table = conn.open_table("chunks").execute().await?;

    // The brief is a generation instruction, not a search query. Distill it to
    // bare Arabic topical keywords before searching; in parallel, extract any
    // named author/source used to bias retrieval (see the scope block below).
    // The two Haiku passes are independent, so run them concurrently.
    let (search_query, scope) = tokio::join!(distill_query(&brief), extract_scope(&brief));
    let search_query = search_query?;
    println!("Brief: {brief}");
    println!("Search query (distilled): {search_query}");

    // Hybrid retrieval: dense (BGE-M3) + BM25 lexical over `text_normalized`,
    // fused by LanceDB's built-in RRF reranker. The dense vector is normalized
    // inside `embed_texts`; the FTS side gets the same normalization so an
    // unvowelled/hamza-variant query matches the folded corpus.
    //
    // Setting BOTH `nearest_to` and `full_text_search` on one builder is the
    // intended hybrid trigger, NOT a bug — don't "fix" it by splitting the call.
    // In lancedb 0.30, `VectorQuery::execute()` checks for a full-text query and,
    // when present, dispatches to `execute_hybrid`: it runs the two retrievers in
    // parallel and fuses them with `RRFReranker::default()` (K=60). A hand-rolled
    // in-memory RRF would only duplicate that exact logic.
    let q_emb = embeddings_model.embed_texts([search_query.clone()]).await?;
    let q_vec: Vec<f32> = q_emb
        .into_iter()
        .next()
        .map(|e| e.vec.into_iter().map(|v| v as f32).collect())
        .unwrap_or_default();

    // `limit` in the hybrid path caps *each* retriever's candidate pool as well
    // as the final cut, so fetch a deep pool for RRF to fuse over, then show only
    // the strongest few.
    const CANDIDATE_POOL: usize = 30;
    const DISPLAY_N: usize = 7;

    // Person/source-aware retrieval. The brief may name an author («عند ابن
    // تيمية») or a book («كما في أخصر المختصرات»), but distillation strips those
    // from the topical query — so we extract the entity separately and use it to
    // *bias* (not hard-filter) retrieval: run the same topical hybrid query once
    // scoped to the entity and once unscoped, then rank in-scope hits first and
    // backfill from the general pool. A brief naming nobody behaves exactly as
    // the topical-only path did. (The extraction itself ran above, alongside
    // distillation; here we just turn it into a retrieval filter.)
    let filter = scope_filter(&scope);
    if let Some(f) = &filter {
        println!(
            "Scope: person={:?} aliases={:?} source={:?}",
            scope.person, scope.aliases, scope.source
        );
        println!("Scope filter: {f}\n");
    }

    println!("--- [ Hybrid Retrieval (dense + BM25, native RRF) ] ---");
    let general = hybrid_search(&table, q_vec.clone(), &search_query, None, CANDIDATE_POOL).await?;
    let scoped = match filter.as_deref() {
        Some(f) => {
            hybrid_search(
                &table,
                q_vec.clone(),
                &search_query,
                Some(f),
                CANDIDATE_POOL,
            )
            .await?
        }
        None => Vec::new(),
    };

    // Boost-with-fallback: in-scope hits first, then the general pool, deduped by
    // chunk_id. When a scope is present and the corpus is rich on it, the whole
    // display is in-scope; when it is thin, general hits fill the remainder.
    let mut seen: HashSet<String> = HashSet::new();
    let mut hits: Vec<(String, String)> = Vec::new();
    for (id, text) in scoped.into_iter().chain(general) {
        if seen.insert(id.clone()) {
            hits.push((id, text));
        }
    }

    let mut top_id: Option<String> = None;
    // Retrieved excerpts captured for grounding (not just printed): each becomes a
    // static context document handed to the generation agent below.
    let mut excerpts: Vec<String> = Vec::new();
    for (i, (id, text)) in hits.iter().take(DISPLAY_N).enumerate() {
        if top_id.is_none() {
            top_id = Some(id.clone());
        }
        let clean = strip_span_tags(text);
        println!("\n[Excerpt #{}] (ID: {id})", i + 1);
        println!("{clean}");
        excerpts.push(clean);
    }
    println!("\n-----------------------------------------\n");

    // Expand context around the top hit. If excerpt #1 is on-topic, the
    // generation model needs the surrounding masʾala, not a lone ~300-char
    // fragment. The page_id band is self-clamping: a hit at the first/last page
    // of the book simply returns fewer neighbors (the `book_id =` clause also
    // stops it bleeding into an adjacent book) — no off-by-one, no error.
    const CONTEXT_WINDOW: i64 = 3; // pages on each side of the hit
    let mut expanded_context = String::new();
    if let Some(top_id) = &top_id {
        println!("--- [ Expanded context around top hit {top_id} (±{CONTEXT_WINDOW} pages) ] ---");
        match expand_context(&table, top_id, CONTEXT_WINDOW).await {
            Ok(ctx) if !ctx.trim().is_empty() => {
                println!("{ctx}");
                expanded_context = ctx;
            }
            Ok(_) => println!("(no neighboring pages found)"),
            Err(e) => eprintln!("context expansion failed: {e}"),
        }
        println!("\n-----------------------------------------\n");
    }

    // Ground the deck in the corpus by handing the model the passages we already
    // retrieved — the hybrid (dense + BM25, native RRF) hits plus the
    // chapter-clamped expansion around the top hit — as static context documents.
    //
    // We use `.context()` rather than `.dynamic_context(N, index)` deliberately:
    // dynamic_context would make rig run its OWN plain-dense search on the raw
    // brief, throwing away the BM25 fusion, the Haiku query distillation, and the
    // chapter clamp. Static context feeds the identical `.documents()` request
    // slot — same delivery to the model — but sourced from our pipeline. These
    // per-brief documents stay uncached; the static SYSTEM_PROMPT and the tool
    // definition remain cached via `with_prompt_caching`.
    let mut builder = AgentBuilder::new(generation_model)
        .preamble(GENERATOR_PROMPT)
        .max_tokens(8000)
        // The strong system prompt makes the model emit exactly one tool call.
        // `tool_choice` is left on Auto so that after the tool builds the deck
        // the model can produce a short closing confirmation — forcing
        // `ToolChoice::Specific` would re-force the tool on every loop turn and
        // never terminate.
        .tool(UpdateBuildContent::new(ppt_gen_dir));

    // The expanded masʾala around the top hit is the richest single passage; the
    // ranked excerpts add breadth across the other retrieved books.
    if !expanded_context.trim().is_empty() {
        builder = builder.context(&expanded_context);
    }
    for ex in &excerpts {
        builder = builder.context(ex);
    }
    let agent = builder.build();

    println!("Generating deck…\n");

    // `max_turns(>=2)`: turn 1 calls the tool (splice + validate + build); the
    // final turn lets the model confirm. Validation/build errors are returned to
    // the model so it can self-correct within the remaining turns.
    //
    // `.extended_details()` switches the response from a bare `String` to a
    // `PromptResponse` carrying the run's aggregated token `usage` and a
    // per-turn `completion_calls` list — that's how we measure where the spend
    // goes (resent context across turns vs. output) and how many turns it
    // actually took (incl. any self-correction after a build error).
    let response = agent.prompt(brief).max_turns(4).extended_details().await?;
    println!("{}", response.output);

    println!("\n--- [ Generation cost ] ---");
    let per_turn: Vec<(usize, rig::completion::Usage)> = response
        .completion_calls()
        .iter()
        .map(|c| (c.call_index, c.usage))
        .collect();
    report_turns(&per_turn);
    report_usage(
        "generator (Sonnet)",
        &usage::Pricing::SONNET_4,
        &response.usage,
        response.requests(),
    );

    Ok(())
}
