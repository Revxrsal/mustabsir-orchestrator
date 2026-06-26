//! Small typed accessors for pulling a named column out of an Arrow
//! `RecordBatch` with a clear error when it is missing or the wrong type.
//! Shared by the retrieval and context-expansion passes.

use arrow::array::{Array, Int32Array, LargeStringArray, ListArray, StringArray};

pub(crate) fn col_i32<'a>(
    batch: &'a arrow::record_batch::RecordBatch,
    name: &str,
) -> anyhow::Result<&'a Int32Array> {
    batch
        .column_by_name(name)
        .and_then(|c| c.as_any().downcast_ref::<Int32Array>())
        .ok_or_else(|| anyhow::anyhow!("missing/!i32 column {name}"))
}

/// Extract a string column as owned values, accepting both `Utf8` and
/// `LargeUtf8`. LanceDB's hybrid-search RRF path upcasts `Utf8` columns to
/// `LargeUtf8` in the fused result, so the downcast must handle both types.
pub(crate) fn col_str(
    batch: &arrow::record_batch::RecordBatch,
    name: &str,
) -> anyhow::Result<Vec<String>> {
    let col = batch
        .column_by_name(name)
        .ok_or_else(|| anyhow::anyhow!("missing/!str column {name}"))?;
    if let Some(arr) = col.as_any().downcast_ref::<StringArray>() {
        return Ok((0..arr.len())
            .map(|i| if arr.is_null(i) { String::new() } else { arr.value(i).to_string() })
            .collect());
    }
    if let Some(arr) = col.as_any().downcast_ref::<LargeStringArray>() {
        return Ok((0..arr.len())
            .map(|i| if arr.is_null(i) { String::new() } else { arr.value(i).to_string() })
            .collect());
    }
    Err(anyhow::anyhow!("missing/!str column {name}"))
}

pub(crate) fn col_list<'a>(
    batch: &'a arrow::record_batch::RecordBatch,
    name: &str,
) -> anyhow::Result<&'a ListArray> {
    batch
        .column_by_name(name)
        .and_then(|c| c.as_any().downcast_ref::<ListArray>())
        .ok_or_else(|| anyhow::anyhow!("missing/!list column {name}"))
}
