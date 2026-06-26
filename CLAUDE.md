# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`mustabsir-orchestrator` is a Rust binary that turns a free-text brief into an
Arabic-first PowerPoint deck. It uses **rig 0.39** (`rig` crate, `reqwest`
feature) to drive an Anthropic _model (`claude-sonnet-4-6`) that, constrained by a
strict system prompt, emits exactly one `update_build_content` tool call. The
tool then shells out to a **separate Python project** (`mustabsir-ppt-gen`) to
render the `.pptx`. This repo contains no PowerPoint logic of its own — it is the
LLM + glue layer; all slide rendering lives in `mustabsir-ppt-gen`.

## Build / run

```bash
cargo check                       # fast type-check
cargo build
cargo run -- "your brief here"    # generate a deck from a brief (any language)
cargo run                         # no brief -> falls back to a sample Arabic topic
```

There are no tests in this crate yet.

### Required environment (`.env`)

Two variables, both read from `.env` at the repo root:

- `ANTHROPIC_API_KEY` — read at **runtime** via `std::env::var`.
- `PPT_GEN_DIR` — absolute path to the sibling `mustabsir-ppt-gen` checkout. **Read
  at compile time** via the `dotenv!` macro (`dotenv_codegen`). Consequence: if
  `PPT_GEN_DIR` is missing or changes, you must edit `.env` and **rebuild** — a
  runtime-only change has no effect, and a missing key is a *compile* error, not a
  runtime one.

`uv` must be on `PATH` (the tool runs `uv run ...` inside `PPT_GEN_DIR`).

## Architecture

### The _model ↔ host contract

The flow is a single rig agent loop (`agent.prompt(brief).max_turns(4)`):

1. The system prompt (`src/system-prompt.md`, embedded via `include_str!`)
   forbids prose and forces one `update_build_content` tool call whose only
   argument is `python_code` — the statements for the body of
   `build_content(deck)` in `mustabsir-ppt-gen`, written at **column 0** using a
   tightly restricted `ArabicDeck`/`Slide`/`TextBlock` helper API.
2. `src/slide_tool.rs` (`UpdateBuildContent`, the `rig::tool::Tool` impl) is the
   host side. Its `call()`:
   - **guards** the code against a denylist (`import`/`from` per line, plus
     `def build_content`, `deck.save`, `.presentation`, `.slide`, `__`);
   - **indents** every non-empty line by 4 spaces (_model emits at column 0);
   - **splices** it into `PPT_GEN_DIR/src/mustabsir_ppt_gen/generate.py`, replacing
     everything between the `EDIT BELOW ... CONTENT ONLY` and
     `EDIT ABOVE ... CONTENT ONLY` marker lines;
   - **validates** the spliced file with `uv run python -c "...ast.parse..."`;
   - **builds** with `uv run mustabsir-ppt-gen -o <output>`.
   - On *any* failure it **restores the original `generate.py`** and returns the
     error string to the _model so it can self-correct on a later turn.

The marker strings in `generate.py` are the integration seam — if they change on
the Python side, `splice()` in `slide_tool.rs` must match (it matches the
substrings `EDIT BELOW` / `EDIT ABOVE` + `CONTENT ONLY`, ignoring indentation).

### Things that are deliberate (don't "fix" them)

- **`tool_choice` is left on Auto, not `Specific`.** rig applies `tool_choice` to
  *every* turn of the agent loop, so forcing the tool would re-force it after the
  build and never terminate (it would hit `MaxTurnsError`). The strict system
  prompt already guarantees the single call.
- **Prompt caching** is enabled by building the _model explicitly:
  `client.completion_model(...).with_prompt_caching()` then
  `AgentBuilder::new(_model)` — *not* `client.agent(name)`, which builds an
  uncached _model. This caches the static system prompt + tool definition (~10%
  cache-read rate within the 5-min TTL). `apply_cache_control` is the low-level
  alternative and is **not** usable from the agent-builder path.

### The Python side (`mustabsir-ppt-gen`, sibling repo)

Rendering, RTL handling, and font embedding live there. Key facts the
orchestrator depends on:

- `build_content(deck: ArabicDeck)` in `src/mustabsir_ppt_gen/generate.py` is the
  splice target; `uv run mustabsir-ppt-gen -o <path>` is the build entrypoint
  (defined as the `mustabsir-ppt-gen` script in its `pyproject.toml`).
- `ArabicDeck.save()` handles RTL/right-alignment, sets the complex-script
  (`<a:cs>`) typeface, and embeds the Amiri (`BODY_FONT`) and KFGQPC Uthman Taha
  Naskh (`QURAN_FONT`) fonts. The _model picks only content; never the plumbing.
- Embedded fonts render reliably in **PowerPoint on Windows**; macOS / Google
  Slides may need the fonts installed locally.

When editing what the _model is allowed to emit, keep three things in sync: the
allowed-API section of `src/system-prompt.md`, the actual `ArabicDeck` API in
`mustabsir-ppt-gen`, and the denylist in `slide_tool.rs`.
