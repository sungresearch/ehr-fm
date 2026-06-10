"""Parquet schema and flush helpers for embedding-mode pretokenized output."""

import os

import pyarrow as pa
import pyarrow.parquet as pq


def _embedding_schema():
    return pa.schema(
        [
            pa.field("subject_id", pa.int64()),
            pa.field("index_time", pa.timestamp("ns")),
            pa.field("embedding_text_ids", pa.list_(pa.int32())),
            pa.field("token_ids", pa.list_(pa.int32())),
            pa.field("numeric_features", pa.list_(pa.list_(pa.float32()))),
            pa.field("age", pa.list_(pa.float32())),
            pa.field("age_normalized", pa.list_(pa.float32())),
            pa.field("length", pa.int32()),
        ]
    )


def _embedding_flush(rows, writer, out_dir, file_name="patients_tokenized.parquet", final=False):
    table = pa.Table.from_pylist(rows, schema=_embedding_schema())
    if writer is None:
        output_path = os.path.join(out_dir, file_name)
        writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
    writer.write_table(table)
    if final:
        writer.close()
        writer = None
    return writer
