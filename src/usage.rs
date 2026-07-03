//! Token-usage + cost instrumentation for the LLM calls.
//!
//! rig aggregates per-turn token [`Usage`] for an agent run once the prompt
//! request is switched into `.extended_details()` mode (the return type then
//! carries `usage` + `completion_calls`). This module pretty-prints that, counts
//! the completion requests (turns), and converts tokens into an approximate
//! dollar cost so we can see where each deck's spend actually goes — the
//! resent-context multiplier on the generator vs. the cheap Haiku passes.
//!
//! Anthropic reports the three input buckets separately: `input_tokens` is the
//! *uncached* portion only, `cached_input_tokens` is read from cache (~0.1x),
//! and `cache_creation_input_tokens` is written to cache (~1.25x). The cost
//! model below mirrors that split, so the estimate stays right even as caching
//! shifts tokens between buckets.

use rig::completion::Usage;

/// Per-million-token prices in USD. **These are estimates — verify against your
/// current Anthropic pricing and edit.** They are the only knob in the cost
/// math; getting them right makes every `~$` figure below correct.
#[derive(Clone, Copy)]
pub struct Pricing {
    pub input_per_mtok: f64,
    pub output_per_mtok: f64,
    pub cache_write_per_mtok: f64,
    pub cache_read_per_mtok: f64,
}

impl Pricing {
    /// Claude Sonnet 4.x — the generation model. Adjust to your billed rates.
    pub const SONNET_4: Pricing = Pricing {
        input_per_mtok: 3.00,
        output_per_mtok: 15.00,
        cache_write_per_mtok: 3.75,
        cache_read_per_mtok: 0.30,
    };

    /// Claude Haiku 4.5 — the distill/scope passes. Adjust to your billed rates.
    pub const HAIKU_4_5: Pricing = Pricing {
        input_per_mtok: 1.00,
        output_per_mtok: 5.00,
        cache_write_per_mtok: 1.25,
        cache_read_per_mtok: 0.10,
    };

    /// Dollar cost of one aggregated [`Usage`], billing each input bucket at its
    /// own rate (uncached input, cache read, cache write) plus output.
    pub fn cost(&self, u: &Usage) -> f64 {
        const M: f64 = 1_000_000.0;
        (u.input_tokens as f64) * self.input_per_mtok / M
            + (u.output_tokens as f64) * self.output_per_mtok / M
            + (u.cached_input_tokens as f64) * self.cache_read_per_mtok / M
            + (u.cache_creation_input_tokens as f64) * self.cache_write_per_mtok / M
    }
}

/// Total input tokens billed across all three buckets (uncached + cache read +
/// cache write) — the real "how big was the prompt we paid for" number.
fn billed_input(u: &Usage) -> u64 {
    u.input_tokens + u.cached_input_tokens + u.cache_creation_input_tokens
}

/// Print a one-line aggregate report for a finished agent run: turn count,
/// per-bucket token totals, and an approximate cost.
pub fn report_usage(label: &str, pricing: &Pricing, usage: &Usage, requests: usize) {
    println!(
        "[usage] {label}: {requests} request(s) | input(billed)={} (uncached={} cache_read={} cache_write={}) output={} | ~${:.4}",
        billed_input(usage),
        usage.input_tokens,
        usage.cached_input_tokens,
        usage.cache_creation_input_tokens,
        usage.output_tokens,
        pricing.cost(usage),
    );
}

/// Print a per-turn breakdown so the resent-context multiplier is visible: each
/// completion request the agent loop made, with its own input/output split.
pub fn report_turns(usage_per_call: &[(usize, Usage)]) {
    for (idx, u) in usage_per_call {
        println!(
            "  turn #{idx}: input(billed)={} (uncached={} cache_read={} cache_write={}) output={}",
            billed_input(u),
            u.input_tokens,
            u.cached_input_tokens,
            u.cache_creation_input_tokens,
            u.output_tokens,
        );
    }
}
