//! Stage 7 — build an Arrow `RecordBatch` from chunks and append to LanceDB.
//!
//! The Arrow schema here is the physical contract for the `chunks` table:
//! `embedding` is `FixedSizeList<Float32, 1024>` and `chapter_path` is
//! `List<Utf8>`; everything else is a scalar column. Field order must match the
//! array order in [`chunks_to_batch`].

use std::sync::Arc;

use arrow::array::{
    ArrayRef, BooleanArray, FixedSizeListArray, Float32Array, Int32Array, Int64Array, ListBuilder,
    StringArray, StringBuilder,
};
use arrow::datatypes::{DataType, Field, FieldRef, Schema, SchemaRef};
use arrow::record_batch::{RecordBatch, RecordBatchIterator};
use lancedb::{Connection, Table};

use crate::ingest::error::WriteError;
use crate::ingest::model::Chunk;
use crate::ingest::pipeline::embed::{normalize_for_embedding, EMBED_DIM};

/// Child field for the `Float32` values inside the `embedding` fixed-size list.
/// Must match what `FixedSizeListArray::try_new` records, or `try_new` fails.
fn embedding_item_field() -> FieldRef {
    Arc::new(Field::new("item", DataType::Float32, true))
}

/// Child field for the `Utf8` values inside `chapter_path`. Must match the field
/// `ListBuilder<StringBuilder>` produces (named `item`, nullable).
fn list_item_field() -> FieldRef {
    Arc::new(Field::new("item", DataType::Utf8, true))
}

/// The physical schema of the `chunks` table.
pub fn chunk_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("chunk_id", DataType::Utf8, false),
        Field::new("book_id", DataType::Int64, false),
        Field::new("page_id", DataType::Int64, false),
        Field::new("page_num", DataType::Int32, false),
        Field::new("sequence_num", DataType::Int32, false),
        Field::new("chunk_idx", DataType::Int32, false),
        Field::new("text", DataType::Utf8, false),
        // Tashkil/span-stripped copy of `text` for the BM25 full-text index, so
        // unvowelled queries match the (heavily vowelled) corpus. Not for display.
        Field::new("text_normalized", DataType::Utf8, false),
        Field::new("title_ar", DataType::Utf8, false),
        Field::new("main_author_name_ar", DataType::Utf8, false),
        Field::new("main_author_death_hijri", DataType::Int32, false),
        Field::new("category_id", DataType::Int32, false),
        Field::new("category_name_ar", DataType::Utf8, false),
        Field::new("madhhab", DataType::Utf8, true),
        Field::new("aqeedah_school", DataType::Utf8, true),
        Field::new("sect", DataType::Utf8, false),
        Field::new("era", DataType::Utf8, false),
        Field::new("canonicity_tier", DataType::Int32, false),
        Field::new("is_critical_edition", DataType::Boolean, false),
        Field::new("chapter_path", DataType::List(list_item_field()), false),
        Field::new("contains_quran", DataType::Boolean, false),
        Field::new("contains_isnad", DataType::Boolean, false),
        Field::new(
            "embedding",
            DataType::FixedSizeList(embedding_item_field(), EMBED_DIM as i32),
            false,
        ),
    ]))
}

/// Build one `RecordBatch` from a slice of chunks (all embeddings must be set).
pub fn chunks_to_batch(chunks: &[Chunk]) -> Result<RecordBatch, WriteError> {
    for c in chunks {
        if c.embedding.len() != EMBED_DIM {
            return Err(WriteError::EmbeddingDim {
                expected: EMBED_DIM,
                got: c.embedding.len(),
            });
        }
    }

    let chunk_id = StringArray::from_iter_values(chunks.iter().map(|c| c.chunk_id.as_str()));
    let book_id = Int64Array::from_iter_values(chunks.iter().map(|c| c.book_id));
    let page_id = Int64Array::from_iter_values(chunks.iter().map(|c| c.page_id));
    let page_num = Int32Array::from_iter_values(chunks.iter().map(|c| c.page_num));
    let sequence_num = Int32Array::from_iter_values(chunks.iter().map(|c| c.sequence_num));
    let chunk_idx = Int32Array::from_iter_values(chunks.iter().map(|c| c.chunk_idx));
    let text = StringArray::from_iter_values(chunks.iter().map(|c| c.text.as_str()));
    let text_normalized =
        StringArray::from_iter_values(chunks.iter().map(|c| normalize_for_embedding(&c.text)));
    let title_ar = StringArray::from_iter_values(chunks.iter().map(|c| c.title_ar.as_str()));
    let author =
        StringArray::from_iter_values(chunks.iter().map(|c| c.main_author_name_ar.as_str()));
    let death = Int32Array::from_iter_values(chunks.iter().map(|c| c.main_author_death_hijri));
    let category_id = Int32Array::from_iter_values(chunks.iter().map(|c| c.category_id));
    let category_name =
        StringArray::from_iter_values(chunks.iter().map(|c| c.category_name_ar.as_str()));
    let madhhab = StringArray::from(chunks.iter().map(|c| c.madhhab.as_deref()).collect::<Vec<_>>());
    let aqeedah =
        StringArray::from(chunks.iter().map(|c| c.aqeedah_school.as_deref()).collect::<Vec<_>>());
    let sect = StringArray::from_iter_values(chunks.iter().map(|c| c.sect.as_str()));
    let era = StringArray::from_iter_values(chunks.iter().map(|c| c.era.as_str()));
    let canon = Int32Array::from_iter_values(chunks.iter().map(|c| c.canonicity_tier));
    let crit = BooleanArray::from(chunks.iter().map(|c| c.is_critical_edition).collect::<Vec<_>>());
    let quran = BooleanArray::from(chunks.iter().map(|c| c.contains_quran).collect::<Vec<_>>());
    let isnad = BooleanArray::from(chunks.iter().map(|c| c.contains_isnad).collect::<Vec<_>>());

    // chapter_path: List<Utf8>
    let mut lb = ListBuilder::new(StringBuilder::new());
    for c in chunks {
        for s in &c.chapter_path {
            lb.values().append_value(s);
        }
        lb.append(true);
    }
    let chapter_path = lb.finish();

    // embedding: contiguous Float32 values wrapped as FixedSizeList.
    let mut floats = Vec::with_capacity(chunks.len() * EMBED_DIM);
    for c in chunks {
        floats.extend_from_slice(&c.embedding);
    }
    let values = Arc::new(Float32Array::from(floats)) as ArrayRef;
    let embedding =
        FixedSizeListArray::try_new(embedding_item_field(), EMBED_DIM as i32, values, None)?;

    let arrays: Vec<ArrayRef> = vec![
        Arc::new(chunk_id),
        Arc::new(book_id),
        Arc::new(page_id),
        Arc::new(page_num),
        Arc::new(sequence_num),
        Arc::new(chunk_idx),
        Arc::new(text),
        Arc::new(text_normalized),
        Arc::new(title_ar),
        Arc::new(author),
        Arc::new(death),
        Arc::new(category_id),
        Arc::new(category_name),
        Arc::new(madhhab),
        Arc::new(aqeedah),
        Arc::new(sect),
        Arc::new(era),
        Arc::new(canon),
        Arc::new(crit),
        Arc::new(chapter_path),
        Arc::new(quran),
        Arc::new(isnad),
        Arc::new(embedding),
    ];

    Ok(RecordBatch::try_new(chunk_schema(), arrays)?)
}

/// Connect to a LanceDB database directory (created if absent).
pub async fn connect(db_path: &str) -> Result<Connection, WriteError> {
    Ok(lancedb::connect(db_path).execute().await?)
}

/// Open the table if it exists, otherwise create it empty with [`chunk_schema`].
pub async fn open_or_create_table(conn: &Connection, name: &str) -> Result<Table, WriteError> {
    let names = conn.table_names().execute().await?;
    if names.iter().any(|n| n == name) {
        Ok(conn.open_table(name).execute().await?)
    } else {
        Ok(conn.create_empty_table(name, chunk_schema()).execute().await?)
    }
}

/// Upsert one batch of rows, keyed on `chunk_id`.
///
/// Merge-insert (not a plain append) makes ingestion **idempotent**: re-running a
/// book replaces its existing chunks instead of duplicating them. This matters
/// because chunks are flushed mid-stream (every `BATCH_SIZE`), so a book that
/// fails partway through `pages.jsonl` has already written its early pages — a
/// retry must not double them. `chunk_id` (`book_id:page_id:chunk_idx`) is
/// deterministic, so the same input always maps to the same row: matched rows are
/// overwritten, new rows inserted.
pub async fn add_batch(table: &Table, batch: RecordBatch) -> Result<(), WriteError> {
    let schema = batch.schema();
    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
    // lancedb 0.30's builder methods take `&mut self`; only `execute` consumes
    // `self`, so configure the binding first, then execute it.
    let mut op = table.merge_insert(&["chunk_id"]);
    op.when_matched_update_all(None);
    op.when_not_matched_insert_all();
    op.execute(Box::new(reader)).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk_with_embedding(id: &str, dim: usize) -> Chunk {
        Chunk {
            chunk_id: id.to_string(),
            book_id: 7,
            page_id: 42,
            page_num: 100,
            sequence_num: 3,
            chunk_idx: 0,
            text: "نص".into(),
            title_ar: "كتاب".into(),
            main_author_name_ar: "مؤلف".into(),
            main_author_death_hijri: 300,
            category_id: 1,
            category_name_ar: "العقيدة".into(),
            madhhab: Some("hanbali".into()),
            aqeedah_school: None,
            sect: "sunni".into(),
            era: "mutaqaddim".into(),
            canonicity_tier: 3,
            is_critical_edition: false,
            chapter_path: vec!["باب".into(), "فصل".into()],
            contains_quran: true,
            contains_isnad: false,
            embedding: vec![0.1; dim],
        }
    }

    #[test]
    fn builds_batch_with_expected_shape() {
        let chunks = vec![
            chunk_with_embedding("7:42:0", EMBED_DIM),
            chunk_with_embedding("7:42:1", EMBED_DIM),
        ];
        let batch = chunks_to_batch(&chunks).unwrap();
        assert_eq!(batch.num_rows(), 2);
        assert_eq!(batch.num_columns(), 23);
        assert_eq!(batch.schema(), chunk_schema());
    }

    #[test]
    fn rejects_wrong_embedding_dim() {
        let chunks = vec![chunk_with_embedding("7:42:0", 8)];
        assert!(matches!(
            chunks_to_batch(&chunks),
            Err(WriteError::EmbeddingDim { expected: EMBED_DIM, got: 8 })
        ));
    }

    /// Read-back sanity check against the `./lancedb` table produced by a real
    /// run (`cargo run --bin shamela-ingest -- --mock-embeddings --skip-enrichment`).
    /// Ignored by default since it depends on that external artifact.
    /// Run with: `cargo test --bin shamela-ingest -- --ignored sanity_query`
    #[tokio::test]
    #[ignore = "requires ./lancedb from a prior real run"]
    async fn sanity_query_produced_table() {
        use arrow::array::{Array, FixedSizeListArray};
        use futures::TryStreamExt;
        use lancedb::query::{ExecutableQuery, QueryBase};

        let conn = lancedb::connect("lancedb").execute().await.unwrap();
        let table = conn.open_table("chunks").execute().await.unwrap();

        let count = table.count_rows(None).await.unwrap();
        println!("chunks row count = {count}");
        assert!(count > 0, "table should not be empty");

        // Carry-forward coverage: how many rows have a non-empty chapter_path.
        let with_chapter = table
            .count_rows(Some("array_length(chapter_path) > 0".to_string()))
            .await
            .unwrap();
        println!("rows with non-empty chapter_path = {with_chapter} / {count}");

        let batches: Vec<RecordBatch> = table
            .query()
            .limit(3)
            .execute()
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        let batch = &batches[0];
        assert_eq!(batch.schema(), chunk_schema(), "round-tripped schema must match");

        let emb = batch
            .column_by_name("embedding")
            .unwrap()
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .unwrap();
        assert_eq!(emb.value_length(), EMBED_DIM as i32, "embedding must be 1024-dim");

        let text = batch
            .column_by_name("text")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert!(!text.value(0).is_empty(), "cleaned text must be non-empty");

        let sect = batch
            .column_by_name("sect")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert!(!sect.value(0).is_empty(), "sect must be populated");
        let madhhab = batch
            .column_by_name("madhhab")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let madhhab0 = if madhhab.is_null(0) {
            "<null>".to_string()
        } else {
            madhhab.value(0).to_string()
        };

        println!(
            "sample: chunk_id={} text_len={} sect={} madhhab={}",
            batch
                .column_by_name("chunk_id")
                .unwrap()
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap()
                .value(0),
            text.value(0).len(),
            sect.value(0),
            madhhab0,
        );
    }
}
