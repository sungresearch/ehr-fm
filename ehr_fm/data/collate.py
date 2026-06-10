"""Collation for the flattened (packed) EHRFM batch layout.

``packed_ehr_collate`` concatenates variable-length patient windows into the flat
token sequence the model consumes, recording ``patient_lengths`` for block-diagonal
attention and ``label_indices`` for the next-token-prediction targets. It is a data
pipeline concern (used as a DataLoader ``collate_fn``), independent of the model
architecture, and defines the batch contract between the data layer and the model.
"""

import torch


def packed_ehr_collate(batch):
    """Combine variable-length windows into the flattened EHRFM layout.

    Output:
        { input_ids, ages, normalized_ages, patient_lengths, label_indices,
          patient_ids, index_times, task: {labels} }  (+ embedding-mode fields)
    """
    lens = [ex["length"] for ex in batch]
    ages = torch.cat([ex["age"] for ex in batch])
    anorm = torch.cat([ex["age_normalized"] for ex in batch])
    lbls = torch.cat([ex["labels"] for ex in batch])
    pids = [ex["patient_id"] for ex in batch]
    idx_t = [ex["index_time"] for ex in batch]

    device = ages.device
    patient_lengths = torch.tensor(lens, dtype=torch.int32, device=device)
    label_indices = torch.nonzero(lbls != -100, as_tuple=False)[:, 0].long()
    patient_ids = torch.tensor(pids, dtype=torch.int64, device=device)
    index_times = torch.tensor(idx_t, dtype=torch.int64, device=device)

    toks = torch.cat([ex["input_ids"] for ex in batch])

    result = {
        "input_ids": toks,
        "ages": ages,
        "normalized_ages": anorm,
        "patient_lengths": patient_lengths,
        "label_indices": label_indices,
        "patient_ids": patient_ids,
        "index_times": index_times,
        "task": {
            "labels": lbls[label_indices],
        },
    }

    # Embedding mode fields
    if "embedding_text_ids" in batch[0]:
        result["embedding_text_ids"] = torch.cat([ex["embedding_text_ids"] for ex in batch])
    if "numeric_features" in batch[0]:
        result["numeric_features"] = torch.cat([ex["numeric_features"] for ex in batch])

    return result
