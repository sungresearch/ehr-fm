import random
import time
from collections.abc import Callable, Iterable

from torch.utils.data import DataLoader, Sampler, Subset
from transformers import Trainer

from ehr_fm.data.dataset import TokenizedDataset
from ehr_fm.data.sampler import TokenBudgetBatchSampler


def sample_indices(
    full_dataset: TokenizedDataset,
    batch_sampler: TokenBudgetBatchSampler,
    k_batches: int,
    ensure_random: bool = True,
) -> Subset:
    """
    Pick enough *row indices* from `full_dataset` to ensure the
    TokenBudgetBatchSampler can construct ≤ k_batches batches.

    Returns a `torch.utils.data.Subset` you can feed straight into a
    vanilla DataLoader (no custom wrapper needed).

    Args:
        full_dataset: The dataset to sample from
        batch_sampler: The batch sampler to use for token budget calculations
        k_batches: Maximum number of batches to create
        ensure_random: If True, uses time-based seeding to ensure different
                      samples across calls. If False, uses current random state.
    """
    if ensure_random:
        seed = int(time.time() * 1000000) % (2**32)
        random.seed(seed)

    # 1) Ask the sampler how many tokens each row contributes
    lengths = batch_sampler.lengths

    # 2) Shuffle row indices once, keep adding until we over-fill ≤ k batches
    indices, total_tokens = [], 0
    shuffled = list(range(len(full_dataset)))
    random.shuffle(shuffled)

    for idx in shuffled:
        length = lengths[idx]
        # Would adding this row exceed k_batches?
        if (total_tokens + length) // batch_sampler.tokens_per_batch >= k_batches:
            break
        indices.append(idx)
        total_tokens += length

    return indices


class FMTrainer(Trainer):
    """
    A custom Hugging Face Trainer:

    1.  **Token Budget Batching:** Uses custom batch samplers (like
        `TokenBudgetBatchSampler`) for training and evaluation, allowing batches
        to be constructed based on a maximum token count rather than a fixed
        number of samples.
    2.  **Custom Collation:** Accepts a specific `collate_fn` suitable for
        EHR-FM model input formats (e.g., `packed_ehr_collate`).
    3.  **Random Subset Evaluation:** Can limit evaluation to a random subset
        of the evaluation dataset, ensuring the subset fits within a specified
        maximum number of batches (`max_eval_batches`) according to the
        token budget sampler.
    4.  **Simplified Loss Computation:** Overrides `compute_loss` slightly to
        interface with EHR-FM models that return loss directly.

    Inherits from `transformers.Trainer` and utilizes its core training and
    evaluation loops.

    Args:
        *args: Positional arguments passed directly to the `transformers.Trainer`
            base class constructor.
        collate_fn (Optional[Callable]): The collation function to use for
            constructing batches. Defaults to None.
        loader_num_workers (int): Number of workers for the DataLoader. Defaults to 4.
        loader_pin_memory (bool): Whether to use pinned memory in the DataLoader.
            Defaults to True.
        loader_persistent_workers (bool): Whether to keep workers persistent
            across epochs. Defaults to True.
        train_batch_size (int): The batch size *per device* for training. Only used
            if `train_batch_sampler` is None. Defaults to 1.
        train_batch_sampler (Optional[Sampler | Iterable]): The custom batch sampler
            for the training DataLoader. If provided, `train_batch_size` is ignored
            by the DataLoader. Defaults to None.
        val_batch_size (int): The batch size *per device* for evaluation. Only used
            if `val_batch_sampler` is None or if `max_eval_batches` is set (where
            it's used for the subset DataLoader). Defaults to 1.
        val_batch_sampler (Optional[Sampler | Iterable]): The custom batch sampler
            for the evaluation DataLoader. Required if `max_eval_batches` is set.
            If provided and `max_eval_batches` is None, `val_batch_size` is ignored
            by the DataLoader. Defaults to None.
        max_eval_batches (Optional[int]): If set, limits evaluation to a random
            subset of the data sized to produce at most this many batches according
            to the `val_batch_sampler`'s token budget. Requires `val_batch_sampler`
            to be a `TokenBudgetBatchSampler`. Defaults to None (evaluate on full dataset).
        ensure_random_eval_sampling (bool): If True and max_eval_batches is set,
            ensures that each evaluation uses a different random subset by using
            time-based seeding. Defaults to True.
        **kwargs: Keyword arguments passed directly to the `transformers.Trainer`
            base class constructor.
    """

    def __init__(
        self,
        *args,
        collate_fn: Callable | None = None,
        loader_num_workers: int = 4,
        loader_pin_memory: bool = True,
        loader_persistent_workers: bool = True,
        train_batch_size: int = 1,
        train_batch_sampler: Sampler | Iterable | None = None,
        val_batch_size: int = 1,
        val_batch_sampler: Sampler | Iterable | None = None,
        max_eval_batches: int | None = None,
        ensure_random_eval_sampling: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._collate_fn = collate_fn
        self._loader_num_workers = loader_num_workers
        self._loader_pin_memory = loader_pin_memory
        self._loader_persistent_workers = loader_persistent_workers
        self._train_batch_size = train_batch_size
        self._train_batch_sampler = train_batch_sampler
        self._val_batch_size = val_batch_size
        self._val_batch_sampler = val_batch_sampler
        self._max_eval_batches = max_eval_batches
        self._ensure_random_eval_sampling = ensure_random_eval_sampling

    def get_train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_sampler=self._train_batch_sampler,
            batch_size=self._train_batch_size,
            num_workers=self._loader_num_workers,
            pin_memory=self._loader_pin_memory,
            persistent_workers=self._loader_persistent_workers,
            collate_fn=self._collate_fn,
        )

    def get_eval_dataloader(self, eval_dataset=None):
        ds = eval_dataset or self.eval_dataset
        subset_sampler = None

        # slice if self._max_eval_steps is set
        if self._max_eval_batches is not None:
            subset_indices = sample_indices(
                ds,
                self._val_batch_sampler,
                self._max_eval_batches,
                ensure_random=self._ensure_random_eval_sampling,
            )
            ds = Subset(ds, subset_indices)

            subset_lengths = [self._val_batch_sampler.lengths[i] for i in subset_indices]
            subset_sampler = TokenBudgetBatchSampler(
                ds,
                tokens_per_batch=self._val_batch_sampler.tokens_per_batch,
                min_patients=self._val_batch_sampler.min_patients,
                shuffle=False,
                lengths=subset_lengths,
            )

            return DataLoader(
                ds,
                batch_sampler=subset_sampler,
                batch_size=self._val_batch_size,
                num_workers=self._loader_num_workers,
                pin_memory=self._loader_pin_memory,
                persistent_workers=self._loader_persistent_workers,
                collate_fn=self._collate_fn,
            )

        return DataLoader(
            ds,
            batch_sampler=subset_sampler or self._val_batch_sampler,
            batch_size=self._val_batch_size,
            num_workers=self._loader_num_workers,
            pin_memory=self._loader_pin_memory,
            persistent_workers=self._loader_persistent_workers,
            collate_fn=self._collate_fn,
        )

    # Make sure compute_loss absorbs extra kwargs
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        loss, outputs = model(inputs, return_logits=False)
        return (loss, outputs) if return_outputs else loss
