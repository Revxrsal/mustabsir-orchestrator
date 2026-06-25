# Ingest Pipeline (`shamela-ingest`)

How a directory of Shamela book exports becomes embedded, searchable rows in the
LanceDB `chunks` table. This document covers **indexing only** (raw text →
index); the query/search path lives in `src/main.rs`.

## Entry point

```
cargo run --release --bin shamela-ingest -- --input <dir> [flags]
```

- `src/bin/shamela-ingest.rs` → `ingest::run()` (`src/ingest/mod.rs`) parses the
  CLI and hands off to `pipeline::run()` (`src/ingest/pipeline/mod.rs`).
- **Discovery** (`discover_books`, `mod.rs:332`): recursively walk `--input`; any
  directory containing `book_metadata.json` is a book (and is not descended
  into). Handles the flat sample layout, a single category, the real
  `<category>/<book>/` layout, and `--input` *being* one book dir.
- **Concurrency**: books are processed on a `JoinSet` capped by `--concurrency`
  (default 4). Every book feeds **one serialized writer task** (`writer_task`,
  `mod.rs:190`) — a single writer avoids racing LanceDB commits.

### Key flags

| flag | effect |
|------|--------|
| `--input <dir>` | root to discover books under (default `book-samples`) |
| `--concurrency <n>` | max books in flight (default 4) |
| `--lancedb <dir>` | database directory (default `lancedb`) |
| `--table <name>` | table name (default `chunks`) |
| `--skip-enrichment` | skip Claude metadata; neutral placeholder enrichment |
| `--mock-embeddings` | deterministic fake vectors (no embedder needed) |
| `--dry-run` | discover + count only, touch nothing |

## Per-book flow (`process_book`, `mod.rs:201`)

```
manifest → metadata → toc → enrichment → [stream pages → clean → chunk → embed → write]
```

1. **Manifest / metadata** (`manifest.rs`, `book.rs`) — load and validate. Books
   with `is_hidden = true` are skipped with a logged note.
2. **TOC** (`toc.rs`) — load `toc.jsonl` into a heading hierarchy; each page maps
   to a root→leaf `chapter_path`.
3. **Enrichment** (`enrich.rs`) — one rig `Extractor` call per book over
   `{title, author, death, category, betaka, first 2000 chars}`. Tries
   **Haiku first; escalates once to Sonnet** if the model's self-reported
   `confidence < 0.7`. Output (`madhhab`, `aqeedah_school`, `sect`,
   `canonicity_tier`, `is_critical_edition`) is cached at
   `cache/enrichments/<book_id>.json` (re-run = delete the file). With
   `--skip-enrichment`, a neutral placeholder is used (`sect=unknown`,
   `canonicity_tier=3`).
4. **Page streaming** (`pages.rs`) — `pages.jsonl` is read **lazily**, one row at
   a time (`serde` streaming deserializer), so a book never fully materializes.
5. **Cleaning** (`pages::clean`) applied per page, in order:
   - `\r` → `\n` (Shamela uses CR as the paragraph delimiter)
   - strip footnote markers `(¬N)` (ASCII or Arabic-Indic digits)
   - strip `<img …>` tags (including base64 data URIs)
   - **preserve** `﴿…﴾`, honorifics, and `<span data-type='title'>` headings
6. **Chunking** (`chunk.rs`) — see below.
7. **Embedding** (`flush`, `mod.rs:307`) — see below.
8. **Write** (`write.rs`) — build an Arrow `RecordBatch` and **upsert** it.

## Chunking strategy (`chunk.rs`)

> Do not change without asking — this is a tuned default.

- **Section split**: each cleaned page is split on `<span data-type="title">`
  heading boundaries. Text before the first heading is its own section.
- **Sliding window**: within each section, a **600-whitespace-token window with
  80-token overlap** (`CHUNK_CAP=600`, `OVERLAP=80`). Token count is
  `split_whitespace().count()` — no tokenizer dependency.
- **Inter-page overlap**: the last `PAGE_OVERLAP` tokens of a page are prepended
  to the next (`tail_tokens`), so a passage straddling a page break is captured
  whole on each side. Overlap never cascades beyond one boundary.
- **`chunk_idx`** increments across the whole page; **`chunk_id`** is the
  deterministic template **`"{book_id}:{page_id}:{chunk_idx}"`**.
- `page_num` is `Option` in the source (`null` on unnumbered front matter) and is
  coerced to the **`-1` sentinel** when built into a `Chunk`, keeping the LanceDB
  column non-nullable `Int32`. Reading order uses `sequence_num`/`page_id`, never
  `page_num`.

## Normalization (`embed.rs`)

`normalize_for_embedding` is applied **symmetrically** — to document text at
ingest and to queries at search time — so the index and queries share one surface
form. Two transforms:

1. **strip `<span>` heading tags** (keeping inner topical text)
2. **remove Arabic tashkil** — harakat, tanwin, superscript alef, Quranic marks —
   and the tatweel elongation `ـ`

> Note: this strips diacritics but does **not** fold orthographic variants
> (`أ/إ/آ → ا`, `ة → ه`, `ى → ي`). See the architecture review for the recall
> implications on the BM25 side.

The stored `text` column keeps the **original** vowelled, span-tagged form (for
display); `text_normalized` stores the normalized copy (for BM25).

## Embedding (`flush`, `mod.rs:307`)

- Each chunk is sent to BGE-M3 as **`"{chapter -> path} -> {text}"`** (`embed_text`,
  `mod.rs:298`). The chapter prefix steers the vector toward the section's topic
  and is **not stored**.
- The embedder (`LocalBgeM3Model` / `Embedder::Http`) POSTs to `$BGE_M3_URL` with
  `{"texts":[…]}` → `{"embeddings":[[…;1024]]}`, applying `normalize_for_embedding`
  internally. Output is one **1024-dim, L2-normalized** vector per chunk.
- `--mock-embeddings` swaps in deterministic seeded unit vectors so the full
  pipeline runs without the embedding service.
- Awaiting `embed` inside the per-book loop provides **backpressure**: a book
  can't race ahead of the embedder.

## Write & index (`write.rs`, `mod.rs:154`)

- **`add_batch` upserts** via `merge_insert(["chunk_id"])`
  (`when_matched_update_all` + `when_not_matched_insert_all`). This makes
  ingestion **idempotent**: re-running a book — even one that failed partway
  through `pages.jsonl` — replaces its chunks instead of duplicating them, because
  `chunk_id` is deterministic.
- After all rows are written (`rows_written > 0`), a single **FTS (BM25) index**
  is built on `text_normalized` (`mod.rs:157`, `replace(true)`).
- **No vector (ANN) index and no distance metric are declared** — dense search is
  currently a flat scan; `nearest_to` defaults to L2 (which ranks identically to
  cosine only because the vectors are normalized).

### Stored schema (`chunk_schema`, `write.rs:35`)

| group | columns |
|-------|---------|
| identity | `chunk_id`, `book_id`, `page_id`, `page_num`, `sequence_num`, `chunk_idx` |
| text | `text`, `text_normalized` |
| bibliography (denormalized) | `title_ar`, `main_author_name_ar`, `main_author_death_hijri`, `category_id`, `category_name_ar` |
| enrichment | `madhhab`, `aqeedah_school`, `sect`, `era`, `canonicity_tier`, `is_critical_edition` |
| chunk signals | `chapter_path` (List<Utf8>), `contains_quran`, `contains_isnad` |
| vector | `embedding` (`FixedSizeList<Float32, 1024>`) |

`era` is derived mechanically from `main_author_death_hijri` (`Era::from_death_hijri`,
`model.rs:254`). `contains_isnad` is currently always `false` (not yet computed).

## Failure handling & resumability

- A book that fails any stage is **logged and appended to `failures.jsonl`**
  (`{book_dir, error}`) and **never crashes the run** (`mod.rs:131`).
- `run_ingest.py` is a resumable wrapper: it writes every discovered book dir to
  `remaining.txt`, runs `shamela-ingest --input <dir>` per book, and removes each
  success from the list — so a crash/Ctrl-C loses at most one book. With the
  `merge_insert` upsert, retries are safe and dedup'd.

## Environment

| var | when read | purpose |
|-----|-----------|---------|
| `BGE_M3_URL` | compile-time (`dotenv!`) | embedding service endpoint |
| `ANTHROPIC_API_KEY` | compile-time (`dotenv!`) | enrichment (Haiku/Sonnet) |

`.env` is also loaded at runtime (`mod.rs:64`) for completeness. Changing a
compile-time value requires a rebuild.
