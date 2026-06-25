//! Small typed accessors for pulling a named column out of an Arrow
//! `RecordBatch` with a clear error when it is missing or the wrong type.
//! Shared by the retrieval and context-expansion passes.

use arrow::array::{Int32Array, ListArray, StringArray};

pub(crate) fn col_i32<'a>(
    batch: &'a arrow::record_batch::RecordBatch,
    name: &str,
) -> anyhow::Result<&'a Int32Array> {
    batch
        .column_by_name(name)
        .and_then(|c| c.as_any().downcast_ref::<Int32Array>())
        .ok_or_else(|| anyhow::anyhow!("missing/!i32 column {name}"))
}

pub(crate) fn col_str<'a>(
    batch: &'a arrow::record_batch::RecordBatch,
    name: &str,
) -> anyhow::Result<&'a StringArray> {
    batch
        .column_by_name(name)
        .and_then(|c| c.as_any().downcast_ref::<StringArray>())
        .ok_or_else(|| anyhow::anyhow!("missing/!str column {name}"))
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
