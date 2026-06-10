"""End-to-end test for compute_representations.create_representations.

The selection of one vector per patient happens via
``end_indices = torch.cumsum(patient_lengths, dim=0) - 1`` inside the real
function. This test actually calls create_representations (with a tiny model and
a synthetic tokenized parquet) and verifies each saved representation equals the
backbone hidden state at that patient's LAST token -- so dropping the ``- 1``
(or otherwise mis-selecting) would fail here.
"""

from datetime import datetime

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from safetensors.torch import save_file as save_safetensors
from torch import nn

from ehr_fm.data import packed_ehr_collate
from ehr_fm.data.dataset import TokenizedDataset
from ehr_fm.logger import setup_logging
from ehr_fm.models import DualPathInputEncoder
from ehr_fm.models.config import EHRFMConfig
from ehr_fm.models.transformer import EHRFM
from ehr_fm.scripts.compute_representations import (
    _load_encoder_weights,
    create_representations,
)

# (subject_id, token_ids) -- distinct lengths exercise the cross-patient boundary.
_PATIENTS = [
    (1, [0, 10, 11, 12, 13]),
    (2, [0, 20, 21]),
    (3, [0, 30, 31, 32]),
]


def _write_tokenized(path):
    schema = pa.schema(
        [
            pa.field("subject_id", pa.int64()),
            pa.field("index_time", pa.timestamp("ns")),
            pa.field("token_ids", pa.list_(pa.int32())),
            pa.field("age", pa.list_(pa.float32())),
            pa.field("length", pa.int32()),
            pa.field("age_normalized", pa.list_(pa.float32())),
        ]
    )
    rows = []
    for i, (sid, tokens) in enumerate(_PATIENTS):
        n = len(tokens)
        rows.append(
            {
                "subject_id": sid,
                "index_time": datetime(2023, 1, i + 1),
                "token_ids": pa.array(tokens, type=pa.int32()),
                "age": pa.array([float(j) for j in range(n)], type=pa.float32()),
                "length": n,
                "age_normalized": pa.array([0.0] * n, type=pa.float32()),
            }
        )
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)


def _tiny_model():
    cfg = EHRFMConfig(
        transformer={
            "vocab_size": 100,
            "hidden_size": 32,
            "intermediate_size": 48,
            "n_heads": 4,
            "n_layers": 2,
            "attention_width": 64,
            "input_mode": "discrete",
        },
        task={"task_type": "sequence_classification", "n_classes": 100},
    )
    return EHRFM(cfg).eval()


def test_create_representations_selects_last_token(tmp_path):
    torch.manual_seed(0)
    tok_path = tmp_path / "tok.parquet"
    out_path = tmp_path / "reps.parquet"
    _write_tokenized(tok_path)
    model = _tiny_model()

    create_representations(
        tokenized_file_path=tok_path,
        model=model,
        device=torch.device("cpu"),
        output_path=out_path,
        max_tokens_per_batch=4096,
        min_patients_per_batch=1,
        shuffle_batches=False,
        loader_num_workers=0,
        logger=setup_logging(),
    )

    out = pl.read_parquet(out_path)
    reps = {row["id"]: torch.tensor(row["representations"]) for row in out.iter_rows(named=True)}

    # One representation per patient, of the right width, all finite.
    assert set(reps) == {str(sid) for sid, _ in _PATIENTS}
    assert all(r.shape == (32,) for r in reps.values())
    assert all(torch.isfinite(r).all() for r in reps.values())

    # Reproduce the backbone forward (block-diagonal attention => per-patient hidden
    # states are independent of batch order) and check the last-token selection.
    dataset = TokenizedDataset(tok_path, max_length=4096, one_window=True)
    batch = packed_ehr_collate([dataset[i] for i in range(len(_PATIENTS))])
    with torch.no_grad():
        hidden = model.transformer(batch)
    end_indices = torch.cumsum(batch["patient_lengths"], dim=0) - 1
    for i, (sid, _) in enumerate(_PATIENTS):
        assert torch.allclose(reps[str(sid)], hidden[end_indices[i]], atol=1e-4)


class TestLoadEncoderWeights:
    @staticmethod
    def _embedding_model(seed):
        torch.manual_seed(seed)
        cfg = EHRFMConfig(
            transformer={
                "vocab_size": 50,
                "hidden_size": 16,
                "intermediate_size": 24,
                "n_heads": 2,
                "n_layers": 2,
                "attention_width": 8,
                "input_mode": "embedding",
                "embedding_dim": 8,
                "use_numerical_path": True,
                "numerical_input_dim": 5,
                "numerical_hidden_dim": 8,
            },
            task={"task_type": "sequence_classification", "n_classes": 50},
        )
        model = EHRFM(cfg)
        text_emb = nn.Embedding(10, 8)
        text_emb.weight.requires_grad = False
        model.transformer.set_input_encoder(
            DualPathInputEncoder(
                text_embedding=text_emb,
                hidden_size=16,
                use_numerical_path=True,
                numerical_input_dim=5,
                numerical_hidden_dim=8,
            )
        )
        return model

    def test_restores_encoder_but_not_text_embedding(self, tmp_path):
        trained = self._embedding_model(seed=0)
        fresh = self._embedding_model(seed=1)

        ckpt = tmp_path / "checkpoint-100"
        ckpt.mkdir()
        save_safetensors(trained.state_dict(), str(ckpt / "model.safetensors"))

        t_proj = trained.transformer.embed.text_projection.mlp[0].weight
        t_num = trained.transformer.embed.numerical_encoder.mlp[0].weight
        f_text_before = fresh.transformer.embed.text_embedding.weight.clone()

        # Different init -> encoder weights differ before loading.
        assert not torch.equal(fresh.transformer.embed.text_projection.mlp[0].weight, t_proj)

        _load_encoder_weights(fresh, ckpt, setup_logging())

        # text_projection / numerical_encoder restored from the checkpoint...
        assert torch.equal(fresh.transformer.embed.text_projection.mlp[0].weight, t_proj)
        assert torch.equal(fresh.transformer.embed.numerical_encoder.mlp[0].weight, t_num)
        # ...but the frozen text_embedding is NOT overwritten.
        assert torch.equal(fresh.transformer.embed.text_embedding.weight, f_text_before)

    def test_missing_checkpoint_is_a_noop(self, tmp_path):
        fresh = self._embedding_model(seed=2)
        before = fresh.transformer.embed.text_projection.mlp[0].weight.clone()
        _load_encoder_weights(fresh, tmp_path, setup_logging())  # empty dir, no safetensors
        assert torch.equal(fresh.transformer.embed.text_projection.mlp[0].weight, before)
