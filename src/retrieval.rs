//! Hybrid (dense + BM25, native RRF) retrieval over the `chunks` table.

use futures::TryStreamExt;
use lance_index::scalar::FullTextSearchQuery;
use lancedb::query::{ExecutableQuery, QueryBase, Select};

use crate::arrow_cols::col_str;
use crate::ingest::pipeline::embed::normalize_for_embedding;

/// One hybrid (dense + BM25, native RRF) retrieval pass, optionally constrained to
/// a scope filter. Returns `(chunk_id, text)` rows in fused-rank order.
pub(crate) async fn hybrid_search(
    table: &lancedb::Table,
    q_vec: Vec<f32>,
    search_query: &str,
    filter: Option<&str>,
    limit: usize,
) -> anyhow::Result<Vec<(String, String)>> {
    let fts = FullTextSearchQuery::new(normalize_for_embedding(search_query));
    let mut q = table
        .query()
        .nearest_to(q_vec)?
        .full_text_search(fts)
        .select(Select::columns(&["chunk_id", "text"]))
        .limit(limit);
    if let Some(f) = filter {
        q = q.only_if(f.to_string());
    }
    let batches: Vec<arrow::record_batch::RecordBatch> =
        q.execute().await?.try_collect().await?;

    let mut out = Vec::new();
    for batch in &batches {
        let ids = col_str(batch, "chunk_id")?;
        let text = col_str(batch, "text")?;
        for i in 0..batch.num_rows() {
            out.push((ids.value(i).to_string(), text.value(i).to_string()));
        }
    }
    Ok(out)
}
