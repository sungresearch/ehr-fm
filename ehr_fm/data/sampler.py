import math

from torch import Generator
from torch.utils.data import BatchSampler, RandomSampler, SequentialSampler

from .dataset import TokenizedDataset


class TokenBudgetBatchSampler(BatchSampler):
    """Creates batches of indices where the total number of tokens does not exceed
    a given budget, while optionally ensuring a minimum number of patients per batch.

    This sampler is useful for training transformer models on sequences of varying
    lengths, allowing for maximization of GPU utilization by packing batches based
    on total tokens rather than a fixed number of sequences (patients).

    Args:
        dataset (TokenizedDataset): The dataset to sample from. It's assumed
            the dataset has `max_length`, `table["length"]`, and `window_index`
            attributes.
        tokens_per_batch (int): The target maximum number of tokens per batch.
            Defaults to 32,768.
        min_patients (int): The minimum number of patients required in a batch
            before it can be yielded due to exceeding the token budget. Defaults to 8.
        drop_last (bool): If True, drops the last batch if it doesn't meet the
            `min_patients` requirement or if it's potentially smaller than other batches.
            If False, the last batch is yielded even if it has fewer than
            `min_patients`. Defaults to False.
        shuffle (bool): If True, shuffles the indices before creating batches.
            Defaults to True.
        lengths (list[int] | None): An optional list of lengths corresponding to
            each index in the dataset. If None, lengths are calculated based on
            the dataset's `window_index` and `max_length`. Defaults to None.

    Note on Overflow Conditions:
        - `tokens_per_batch` can be exceeded: This happens if adding a new sample
          (patient window) to the current batch *would* push the total tokens over
          the limit, BUT the batch hasn't yet reached `min_patients`. In this case,
          the sample is added anyway to try and meet `min_patients`. It can also
          be exceeded if a single sample's length is greater than `tokens_per_batch`.
        - Batches smaller than `min_patients`: This can only occur for the *last*
          batch produced, and only if `drop_last` is set to `False`.
    """

    def __init__(
        self,
        dataset: TokenizedDataset,
        tokens_per_batch: int = 32_768,
        min_patients: int = 8,
        drop_last: bool = False,
        shuffle: bool = True,
        seed: int | Generator = None,
        lengths: list[int] | None = None,
    ):
        # lengths list needs to be built based on the window_index instead of the table
        self.lengths = lengths or [
            min(dataset.max_length, dataset.table["length"][row] - offset)
            for row, offset in dataset.window_index
        ]

        self.tokens_per_batch = tokens_per_batch
        self.min_patients = min_patients
        self.drop_last = drop_last
        self.shuffle = shuffle

        if self.shuffle:
            self.generator = Generator()
            if seed:
                if isinstance(seed, int):
                    self.generator.manual_seed(seed)
                elif isinstance(seed, Generator):
                    self.generator = seed
                else:
                    raise ValueError(
                        f"Expected 'seed' of type int or torch.Generator, " f"but got type '{type(seed)}'"
                    )
            self.sampler = RandomSampler(dataset, replacement=False, generator=self.generator)
        else:
            self.sampler = SequentialSampler(dataset)

    def __iter__(self):
        pool = list(self.sampler)

        batch, tokens = [], 0
        for idx in pool:
            length = self.lengths[idx]
            # Will we overflow?
            if (tokens + length > self.tokens_per_batch) and (len(batch) >= self.min_patients):
                yield batch
                batch, tokens = [], 0
            batch.append(idx)
            tokens += length

        if batch and (not self.drop_last or len(batch) >= self.min_patients):
            yield batch

    def __len__(self):
        # Rough estimate; not used by training loops other than progress bars
        total_tokens = sum(self.lengths)
        return math.ceil(total_tokens / self.tokens_per_batch)
