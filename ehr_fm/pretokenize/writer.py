"""Parquet schema and flush helpers for pretokenized output."""

import os

import pyarrow as pa
import pyarrow.parquet as pq


def _pretokenize_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("subject_id", pa.int64()),
            pa.field("index_time", pa.timestamp("ns")),
            pa.field("token_ids", pa.list_(pa.int32())),
            pa.field("age", pa.list_(pa.float32())),
            pa.field("length", pa.int32()),
            pa.field("age_normalized", pa.list_(pa.float32())),
        ]
    )


def _pretokenize_flush(rows, writer, out_dir, file_name="patients_tokenized.parquet", final=False):
    table = pa.Table.from_pylist(rows, schema=_pretokenize_schema())
    if writer is None:
        output_path = os.path.join(out_dir, file_name)
        writer = pq.ParquetWriter(output_path, table.schema, compression="zstd", use_dictionary=True)

    writer.write_table(table)

    # Explicitly clean up table to free memory immediately
    del table

    if final:
        writer.close()
        writer = None

    return writer
