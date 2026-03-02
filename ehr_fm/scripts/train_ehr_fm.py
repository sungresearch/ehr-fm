import argparse
import dataclasses
import os
import sys
from pathlib import Path

import torch
from torch.optim import AdamW
from transformers import EarlyStoppingCallback, TrainingArguments, get_wsd_schedule
from transformers.trainer_pt_utils import get_parameter_names

from ehr_fm.callbacks import TopKCheckpointCallback, VariableSaveFrequencyCallback
from ehr_fm.cli_utils import normalize_resume_checkpoint, str_to_bool
from ehr_fm.data import TokenBudgetBatchSampler, TokenizedDataset
from ehr_fm.io import read_json_yaml
from ehr_fm.logger import setup_logging
from ehr_fm.models import EHRFM, packed_ehr_collate
from ehr_fm.models.config import EHRFMConfig
from ehr_fm.trainer import FMTrainer


def build_optimizer(config: dict, model: torch.nn.Module, logger) -> torch.optim.Optimizer:
    """Build an optimizer from *config* (keys: name, lr, betas, eps, weight_decay)."""
    optimizer_name = config["name"]

    # Separate parameters for weight decay. We don't want to apply decay to biases, LayerNorm weights, or embeddings.
    decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name and "embed" not in name]

    params_to_optimize = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters and p.requires_grad],
            "weight_decay": config["weight_decay"],
        },
        {
            "params": [
                p for n, p in model.named_parameters() if n not in decay_parameters and p.requires_grad
            ],
            "weight_decay": 0.0,
        },
    ]

    common_args = {
        "lr": config["lr"],
        "betas": list(config["betas"]),
        "eps": config["eps"],
    }

    if optimizer_name == "adamw":
        logger.info(
            f"INFO: Using torch.optim.AdamW. LR: {common_args['lr']}, "
            f"WD: {config['weight_decay']} (applied selectively)."
        )
        return AdamW(params_to_optimize, **common_args)

    elif optimizer_name == "stableadamw":
        from optimi import StableAdamW

        logger.info(
            f"INFO: Using optimi.StableAdamW. LR: {common_args['lr']}, "
            f"WD: {config['weight_decay']} (applied selectively)."
        )
        logger.info(
            "INFO: StableAdamW is configured to decouple the learning rate from "
            "the weight decay. The default weight decay of 0.01 will likely "
            "need to be reduced as the learning rate will not modify the "
            "effective weight decay."
        )

        return StableAdamW(params_to_optimize, decouple_lr=True, **common_args)

    else:
        raise ValueError(f"Unsupported optimizer name: {optimizer_name}. Choose 'adamw' or 'stableadamw'.")


def build_lr_scheduler(config: dict, optimizer: torch.optim.Optimizer, logger):
    """Build an LR scheduler from *config* (key ``name`` selects the schedule type)."""
    scheduler_name = config.get("name")

    if scheduler_name == "wsd":
        logger.info("Using 'wsd' (Warmup-Stable-Decay) scheduler.")
        required_keys = ["num_warmup_steps", "num_decay_steps", "num_training_steps", "decay_type"]

        # Check for required parameters
        missing_keys = [key for key in required_keys if config.get(key) is None]
        if missing_keys:
            raise ValueError(f"For 'wsd' scheduler, you must provide: {', '.join(missing_keys)}")

        return get_wsd_schedule(
            optimizer,
            num_warmup_steps=config["num_warmup_steps"],
            num_decay_steps=config["num_decay_steps"],
            num_training_steps=config["num_training_steps"],
            decay_type=config["decay_type"],
        )
    elif scheduler_name is None:
        logger.info(
            "No custom scheduler specified. Trainer will use default scheduler from TrainingArguments."
        )
        return None
    else:
        raise ValueError(
            f"Unsupported scheduler name: {scheduler_name}. Choose 'wsd' or leave empty for default."
        )


def prepare_data(
    train_path,
    val_path,
    max_tokens_per_batch,
    min_patients_per_batch,
    max_seq_length_per_patient,
    token_dropout_prob,
    min_patient_length,
    stride,
    num_ntp_classes,
    *,
    convert_ages_to_positions: bool = False,
):
    if max_seq_length_per_patient is None:
        max_seq_length_per_patient = max_tokens_per_batch // min_patients_per_batch

    train_dataset = TokenizedDataset(
        train_path,
        max_length=max_seq_length_per_patient,
        one_window=False,
        dropout_prob=token_dropout_prob,
        min_length=min_patient_length,
        stride=stride,
        num_ntp_classes=num_ntp_classes,
        convert_ages_to_positions=convert_ages_to_positions,
    )

    train_sampler = TokenBudgetBatchSampler(
        train_dataset, tokens_per_batch=max_tokens_per_batch, min_patients=min_patients_per_batch
    )

    val_dataset = TokenizedDataset(
        val_path,
        max_length=max_seq_length_per_patient,
        one_window=False,
        dropout_prob=token_dropout_prob,
        num_ntp_classes=num_ntp_classes,
        convert_ages_to_positions=convert_ages_to_positions,
    )

    val_sampler = TokenBudgetBatchSampler(
        val_dataset, tokens_per_batch=max_tokens_per_batch, min_patients=min_patients_per_batch
    )

    return train_dataset, train_sampler, val_dataset, val_sampler


def _detect_positions_mode(dataset, logger) -> bool:
    """Heuristic: ages are positions (0..N-1) and age_normalized are zeros.

    Returns True if detection succeeds on the first available sample; else False.
    """
    try:
        if len(dataset) == 0:
            return False
        sample = dataset[0]
        ages = sample.get("age")
        ages_norm = sample.get("age_normalized")
        if ages is None or ages_norm is None:
            return False
        # Ensure tensors
        try:
            ages = ages.float()
            ages_norm = ages_norm.float()
            if ages.numel() < 2:
                return False
            diffs = ages[1:] - ages[:-1]
            is_unit_step = torch.allclose(diffs, torch.ones_like(diffs), atol=1e-4, rtol=0.0)
            is_norm_zero = ages_norm.abs().sum().item() == 0.0
            return bool(is_unit_step and is_norm_zero)
        except Exception:
            return False
    except Exception as e:
        logger.warning(f"Positions-mode detection skipped due to error: {e}")
        return False


def prepare_model(args, device, logger):
    transformer_config_dict = {
        "vocab_size": args.vocab_size,
        "hidden_size": args.hidden_size,
        "n_layers": args.n_layers,
        "n_heads": args.n_heads,
        "attention_width": args.attention_width,
        "use_normed_ages": args.use_normed_ages,
        "intermediate_size": args.intermediate_size,
        "use_bias": args.use_bias,
        "hidden_act": args.hidden_act,
        "remove_first_block_norm": args.remove_first_block_norm,
        "alternating_dense_layers": args.alternating_dense_layers,
        "dense_every_n_layers": args.dense_every_n_layers,
        "rope_base_sparse": args.rope_base_sparse,
        "rope_base_global": args.rope_base_global,
        "separate_rope_by_attention": args.separate_rope_by_attention,
    }

    cfg = EHRFMConfig(
        transformer=transformer_config_dict,
        task=dict(
            task_type="sequence_classification",
            n_classes=args.num_ntp_classes,
        ),
    )

    if args.initial_weights_path:
        logger.info(f"Initializing model from pretrained weights: {args.initial_weights_path}")
        # We use from_pretrained but enforce our config.
        # This ensures we use the architecture defined by our args/config,
        # not the config.json in the checkpoint directory (if it exists).
        # We set ignore_mismatched_sizes=False (default) to enforce strict loading.
        model = EHRFM.from_pretrained(args.initial_weights_path, config=cfg)
    else:
        logger.info("Initializing model with random weights.")
        model = EHRFM(cfg)

    model = model.to(device)
    model.train()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Total parameters: {total_params}")
    logger.info(f"Trainable parameters: {trainable_params}")
    return model


def train_model(args, model, train_dataset, train_sampler, val_dataset, val_sampler, logger):
    if dataclasses.is_dataclass(TrainingArguments):
        training_arg_names = set(TrainingArguments.__dataclass_fields__.keys())
    else:
        logger.error("TrainingArguments is not a dataclass. Cannot dynamically get fields.")
        training_arg_names = set()

    training_args_dict = {key: value for key, value in vars(args).items() if key in training_arg_names}

    if training_args_dict.get("output_dir") is None:
        raise ValueError("output_dir must be specified for TrainingArguments")

    # Inject lr_scheduler_kwargs for cosine_with_min_lr if provided via config
    if training_args_dict.get("lr_scheduler_type") == "cosine_with_min_lr":
        if "lr_scheduler_kwargs" in training_arg_names:
            scheduler_kwargs = getattr(args, "lr_scheduler_kwargs", None) or {}
            # Map legacy/top-level keys if present in the config
            if (
                "min_lr_rate" not in scheduler_kwargs
                and hasattr(args, "min_lr_ratio")
                and getattr(args, "min_lr_ratio") is not None
            ):
                scheduler_kwargs["min_lr_rate"] = getattr(args, "min_lr_ratio")
            if (
                "min_lr" not in scheduler_kwargs
                and hasattr(args, "min_lr")
                and getattr(args, "min_lr") is not None
            ):
                scheduler_kwargs["min_lr"] = getattr(args, "min_lr")
            if scheduler_kwargs:
                training_args_dict["lr_scheduler_kwargs"] = scheduler_kwargs
                logger.info(f"Passing lr_scheduler_kwargs to TrainingArguments: {scheduler_kwargs}")
        else:
            logger.warning(
                "TrainingArguments on this Transformers version does not support 'lr_scheduler_kwargs'. "
                "'cosine_with_min_lr' requires min_lr or min_lr_rate and may fail without it."
            )

    # Set up HuggingFace's built-in metric tracking for top-K checkpointing
    if args.save_top_k > 0:
        training_args_dict["metric_for_best_model"] = args.metric_for_best_model
        training_args_dict["greater_is_better"] = args.greater_is_better
        training_args_dict["load_best_model_at_end"] = args.load_best_model_at_end

        # Warn if load_best_model_at_end is disabled when using top-K checkpointing
        if not args.load_best_model_at_end:
            logger.warning(
                "Top-K checkpointing works best with load_best_model_at_end=True for HF metric tracking"
            )

        # Ensure evaluation is enabled for top-K checkpointing to work effectively
        if training_args_dict.get("evaluation_strategy") == "no":
            logger.warning("Top-K checkpointing enabled but evaluation_strategy is 'no'. Setting to 'steps'.")
            training_args_dict["evaluation_strategy"] = "steps"

        # Ensure eval_steps is reasonable for responsive top-K checkpointing
        if (
            training_args_dict.get("evaluation_strategy") == "steps"
            and training_args_dict.get("eval_steps", 0) > 1000
        ):
            logger.warning(
                f"eval_steps={training_args_dict.get('eval_steps')} is quite large for "
                "top-K checkpointing. Consider using smaller values (e.g., 100-500) for "
                "more responsive checkpointing."
            )

    # Set up HuggingFace's built-in metric tracking for early stopping
    if args.early_stopping_patience > 0:
        training_args_dict["metric_for_best_model"] = args.metric_for_best_model
        training_args_dict["greater_is_better"] = args.greater_is_better
        training_args_dict["load_best_model_at_end"] = args.load_best_model_at_end

        # Ensure evaluation is enabled for early stopping to work
        if training_args_dict.get("evaluation_strategy") == "no":
            logger.warning("Early stopping enabled but evaluation_strategy is 'no'. Setting to 'steps'.")
            training_args_dict["evaluation_strategy"] = "steps"

        # Ensure eval_steps is reasonable for responsive early stopping
        if (
            training_args_dict.get("evaluation_strategy") == "steps"
            and training_args_dict.get("eval_steps", 0) > 1000
        ):
            logger.warning(
                f"eval_steps={training_args_dict.get('eval_steps')} is quite large for early "
                "stopping. Consider using smaller values (e.g., 100-500) for more responsive "
                "early stopping."
            )

        # Ensure load_best_model_at_end is enabled for early stopping to work properly
        if not args.load_best_model_at_end:
            logger.warning(
                "Early stopping works best with load_best_model_at_end=True to properly track the best metric."
            )

    logger.info(f"Initializing TrainingArguments with: {training_args_dict}")
    training_args_obj = TrainingArguments(**training_args_dict)

    # Build the optimizer
    optimizer_config = {
        "name": args.optimizer_name,
        "lr": args.learning_rate,
        "betas": (args.adam_beta1, args.adam_beta2),
        "eps": args.adam_epsilon,
        "weight_decay": args.weight_decay,
    }
    logger.info(f"Building optimizer ({args.optimizer_name}) with config: {optimizer_config}")
    optimizer = build_optimizer(config=optimizer_config, model=model, logger=logger)

    # Build the LR scheduler
    lr_scheduler = None
    if args.lr_scheduler_name:
        scheduler_config = {
            "name": args.lr_scheduler_name,
            "num_warmup_steps": args.warmup_steps,
            "num_training_steps": args.max_steps,
            "num_decay_steps": args.wsd_num_decay_steps,
            "decay_type": args.wsd_decay_type,
        }
        logger.info(f"Building LR scheduler ({args.lr_scheduler_name}) with config: {scheduler_config}")
        lr_scheduler = build_lr_scheduler(config=scheduler_config, optimizer=optimizer, logger=logger)

    # Normalise mixed bool / string values from YAML configs and CLI.
    resume_checkpoint = normalize_resume_checkpoint(args.resume_from_checkpoint)
    if resume_checkpoint is True:
        # Auto-detect checkpoint to resume from
        output_dir = args.output_dir

        # First, try to find regular HuggingFace checkpoints
        regular_checkpoints = []
        if os.path.exists(output_dir):
            for item in os.listdir(output_dir):
                if item.startswith("checkpoint-") and not item.startswith("checkpoint-topk-"):
                    checkpoint_path = os.path.join(output_dir, item)
                    if os.path.isdir(checkpoint_path):
                        try:
                            step = int(item.split("-")[1])
                            regular_checkpoints.append((step, checkpoint_path))
                        except (ValueError, IndexError):
                            pass

        if regular_checkpoints:
            # Use the latest regular checkpoint
            latest_regular = max(regular_checkpoints, key=lambda x: x[0])
            resume_checkpoint = latest_regular[1]
            logger.info(f"Resuming from regular checkpoint: {resume_checkpoint}")
        else:
            # No regular checkpoints found, try top-K checkpoints
            topk_checkpoints = []
            if os.path.exists(output_dir):
                for item in os.listdir(output_dir):
                    if item.startswith("checkpoint-topk-"):
                        checkpoint_path = os.path.join(output_dir, item)
                        if os.path.isdir(checkpoint_path):
                            try:
                                step = int(item.split("-")[2])  # checkpoint-topk-{step}
                                topk_checkpoints.append((step, checkpoint_path))
                            except (ValueError, IndexError):
                                pass

            if topk_checkpoints:
                # Use the latest top-K checkpoint
                latest_topk = max(topk_checkpoints, key=lambda x: x[0])
                resume_checkpoint = latest_topk[1]
                logger.info(
                    f"No regular checkpoints found. Resuming from top-K checkpoint: {resume_checkpoint}"
                )
            else:
                # No checkpoints found at all
                resume_checkpoint = None
                logger.info("No checkpoints found in output directory. Starting training from scratch.")

    trainer = FMTrainer(
        args=training_args_obj,
        model=model,
        train_dataset=train_dataset,
        train_batch_sampler=train_sampler,
        eval_dataset=val_dataset,
        val_batch_sampler=val_sampler,
        max_eval_batches=args.max_eval_batches,
        collate_fn=packed_ehr_collate,
        optimizers=(optimizer, lr_scheduler),  # Pass the custom optimizer
    )

    # Add top-K checkpointing callback if enabled
    if args.save_top_k > 0:
        logger.info(
            f"Adding TopKCheckpointCallback: save_top_k={args.save_top_k}, "
            f"metric={args.metric_for_best_model}"
        )
        top_k_callback = TopKCheckpointCallback(
            save_top_k=args.save_top_k,
            metric_name=args.metric_for_best_model,
            greater_is_better=args.greater_is_better,
            delete_checkpoint_callback=not args.save_best_checkpoint_only,
            skip_first_n_steps=args.skip_first_n_steps,
        )
        top_k_callback.setup_trainer(trainer)
        trainer.add_callback(top_k_callback)
    else:
        top_k_callback = None

    # Add early stopping callback if enabled
    if args.early_stopping_patience > 0:
        logger.info(
            f"Adding EarlyStoppingCallback: patience={args.early_stopping_patience}, "
            f"metric={args.metric_for_best_model}, threshold={args.early_stopping_threshold}"
        )

        # Ensure evaluation is enabled for early stopping to work
        if training_args_obj.eval_strategy == "no":
            logger.warning(
                "Early stopping enabled but eval_strategy is 'no'. Early stopping requires "
                "evaluation to monitor metrics."
            )
            logger.warning(
                "Consider setting --eval_strategy to 'steps' or 'epoch' for early stopping to work properly."
            )

        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_threshold=args.early_stopping_threshold,
        )
        trainer.add_callback(early_stopping_callback)
        logger.info(
            f"Early stopping will monitor '{args.metric_for_best_model}' with patience={args.early_stopping_patience}"
        )
    else:
        early_stopping_callback = None

    # Add variable save frequency callback if configured
    if (args.early_save_every is not None and args.early_save_every > 0) or (
        args.late_save_every is not None and args.late_save_every > 0
    ):
        logger.info(
            f"Adding VariableSaveFrequencyCallback: early_until={args.early_save_until_step}, "
            f"early_every={args.early_save_every}, late_every={args.late_save_every}"
        )
        variable_save_callback = VariableSaveFrequencyCallback(
            early_save_until_step=args.early_save_until_step,
            early_save_every=args.early_save_every,
            late_save_every=args.late_save_every,
        )
        trainer.add_callback(variable_save_callback)

    logger.info("Starting training...")
    train_output = trainer.train(resume_from_checkpoint=resume_checkpoint)
    logger.info(f"Training finished. Output: {train_output}")

    # Log best checkpoint info if using top-K
    if top_k_callback:
        best_checkpoint = top_k_callback.get_best_checkpoint_path()
        best_metric = top_k_callback.get_best_metric_value()
        if best_checkpoint and best_metric is not None:
            logger.info(f"Best checkpoint: {best_checkpoint}")
            logger.info(f"Best {args.metric_for_best_model}: {best_metric:.6f}")

    # Log early stopping info if training was stopped early
    if early_stopping_callback and hasattr(trainer.state, "log_history") and trainer.state.log_history:
        logger.info("Training completed. Check logs to see if early stopping was triggered.")

    return trainer


def wandb_init(args, logger):
    import wandb

    logger.info("Initializing Weights and Biases Logging")
    wandb.init(
        dir=os.path.join(args.output_dir, "wandb"),
        name=args.run_name,
        project=args.project_name,
        tags=args.tags,
    )
    # Log all arguments as hyperparameters
    wandb.config.update(vars(args))


def main():
    # Main parser definition (includes the positional 'config' argument and all --flags)
    parser = argparse.ArgumentParser(description="Train a EHRFM model.", conflict_handler="resolve")
    parser.add_argument("config", type=Path, help="Path to a YAML/JSON configuration file.")

    # Data arguments
    parser.add_argument("--train_path", type=str, default=None, help="Path to the training data. (Required)")
    parser.add_argument("--val_path", type=str, default=None, help="Path to the validation data. (Required)")
    parser.add_argument("--max_tokens_per_batch", type=int, default=16_384, help="Maximum tokens per batch.")
    parser.add_argument("--min_patients_per_batch", type=int, default=2, help="Minimum patients per batch.")
    parser.add_argument(
        "--min_patient_length", type=int, default=2, help="Minimum length of a patient's sequence."
    )
    parser.add_argument("--stride", type=int, default=None, help="Stride for sliding windows.")
    parser.add_argument(
        "--token_dropout_prob",
        type=float,
        default=0.0,
        help="Probability of dropping tokens during training data preparation.",
    )
    parser.add_argument(
        "--max_seq_length_per_patient",
        type=int,
        default=None,
        help=(
            "Maximum sequence length per patient. "
            "Calculated if None based on max_tokens_per_batch // min_patients_per_batch."
        ),
    )
    parser.add_argument(
        "--convert_ages_to_positions",
        action="store_true",
        help=(
            "Convert ages to sequential positions (0, 1, 2, ...) at runtime. "
            "This makes age behave like token positions instead of actual age values. "
            "Automatically disables use_normed_ages."
        ),
    )

    # Model arguments
    model_args = parser.add_argument_group("Model Arguments")
    model_args.add_argument("--vocab_size", type=int, default=98_304, help="Vocabulary size.")
    model_args.add_argument(
        "--initial_weights_path",
        type=str,
        default=None,
        help="Path to initial model weights (safetensors or bin) to load.",
    )
    model_args.add_argument(
        "--num_ntp_classes", type=int, default=8_192, help="Number of classes for next token prediction."
    )
    model_args.add_argument("--hidden_size", type=int, default=768, help="Hidden size of the model.")
    model_args.add_argument("--n_layers", type=int, default=12, help="Number of layers in the model.")
    model_args.add_argument("--n_heads", type=int, default=12, help="Number of attention heads.")
    model_args.add_argument("--attention_width", type=int, default=496, help="Attention width.")
    model_args.add_argument(
        "--use_normed_ages",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to use normed ages. Use --no-use_normed_ages to disable.",
    )
    model_args.add_argument(
        "--intermediate_size",
        type=int,
        default=1152,
        help="Size of the FFN in the transformer layers (DenseTransformerConfig default: 1152).",
    )
    model_args.add_argument(
        "--use_bias",
        action="store_true",
        help="Whether to use bias terms in the transformer layers (DenseTransformerConfig default: False).",
    )
    model_args.add_argument(
        "--hidden_act",
        type=str,
        default="swiglu",
        help="Activation function in the transformer (DenseTransformerConfig default: 'swiglu').",
    )
    model_args.add_argument(
        "--remove_first_block_norm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Whether to remove the redundant normalization layer from the first block "
            "(DenseTransformerConfig default: True). Use --no-remove_first_block_norm to disable."
        ),
    )
    model_args.add_argument(
        "--alternating_dense_layers",
        action="store_true",
        help="Whether to use alternating dense layers (DenseTransformerConfig default: False).",
    )
    model_args.add_argument(
        "--dense_every_n_layers",
        type=int,
        default=3,
        help="Frequency of dense layers if alternating_dense_layers is True (DenseTransformerConfig default: 3).",
    )
    model_args.add_argument(
        "--rope_base_sparse",
        type=float,
        default=100.0,
        help=(
            "Base value for RoPE frequency in sparse attention layers. "
            "With linspace(0,2), effective theta = base^2. Default 100 gives theta ~10K (standard RoPE). "
            "(DenseTransformerConfig default: 100.0)"
        ),
    )
    model_args.add_argument(
        "--rope_base_global",
        type=float,
        default=10000.0,
        help=(
            "Base value for RoPE frequency in global attention layers. "
            "Default 10000 gives theta ~100M (extended range for long sequences). "
            "(DenseTransformerConfig default: 10000.0)"
        ),
    )
    model_args.add_argument(
        "--separate_rope_by_attention",
        action="store_true",
        help=(
            "If True, use different RoPE configurations for sparse vs global attention. "
            "If False, use rope_base_global for all layers (backward compatible). "
            "(DenseTransformerConfig default: False)"
        ),
    )

    training_args_group = parser.add_argument_group("HuggingFace Training Arguments")
    training_args_group.add_argument(
        "--output_dir", type=str, default=None, help="Output directory for checkpoints and logs. (Required)"
    )
    training_args_group.add_argument(
        "--logging_strategy",
        type=str,
        default="steps",
        choices=["no", "epoch", "steps"],
        help="Logging strategy.",
    )
    training_args_group.add_argument(
        "--logging_steps",
        type=int,
        default=50,
        help="Log every X updates steps if logging_strategy is 'steps'.",
    )
    training_args_group.add_argument(
        "--evaluation_strategy",
        type=str,
        default="steps",
        choices=["no", "epoch", "steps"],
        help="Evaluation strategy.",
    )
    training_args_group.add_argument(
        "--eval_steps",
        type=int,
        default=50,
        help="Evaluate every X updates steps if evaluation_strategy is 'steps'.",
    )
    training_args_group.add_argument(
        "--max_steps", type=int, default=1_000_000, help="Total number of training steps to perform."
    )
    training_args_group.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    training_args_group.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Batch size per device during training. Should be 1 when using TokenBudgetBatchSampler.",
    )
    training_args_group.add_argument(
        "--max_eval_batches", type=int, default=None, help="Maximum number of evaluation batches to run."
    )

    training_args_group.add_argument(
        "--learning_rate", type=float, default=5e-5, help="Initial learning rate."
    )
    training_args_group.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to apply.")
    training_args_group.add_argument(
        "--adam_beta1", type=float, default=0.9, help="Beta1 for Adam optimizer."
    )
    training_args_group.add_argument(
        "--adam_beta2", type=float, default=0.999, help="Beta2 for Adam optimizer."
    )
    training_args_group.add_argument(
        "--adam_epsilon", type=float, default=1e-8, help="Epsilon for Adam optimizer."
    )
    training_args_group.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm.")
    training_args_group.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="linear",
        choices=["linear", "cosine", "constant", "constant_with_warmup", "cosine_with_min_lr"],
        help="Learning rate scheduler type.",
    )
    training_args_group.add_argument(
        "--warmup_steps", type=int, default=0, help="Linear warmup over warmup_steps."
    )
    training_args_group.add_argument(
        "--save_strategy", type=str, default="steps", choices=["no", "epoch", "steps"], help="Save strategy."
    )
    training_args_group.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every X updates steps if save_strategy is 'steps'.",
    )
    training_args_group.add_argument(
        "--save_total_limit",
        type=int,
        default=None,
        help="Limit the total number of checkpoints. Deletes the older checkpoints.",
    )
    training_args_group.add_argument(
        "--fp16", type=str_to_bool, default=False, help="Whether to use 16-bit (mixed) precision training."
    )
    training_args_group.add_argument(
        "--fp16_full_eval",
        type=str_to_bool,
        default=False,
        help="Whether to use full 16-bit precision for evaluation (if fp16 is enabled).",
    )
    training_args_group.add_argument(
        "--bf16", type=str_to_bool, default=False, help="Whether to use bfloat16 precision training."
    )
    training_args_group.add_argument(
        "--bf16_full_eval",
        type=str_to_bool,
        default=False,
        help="Whether to use full bfloat16 precision for evaluation (if bf16 is enabled).",
    )
    training_args_group.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="Number of subprocesses to use for data loading.",
    )
    training_args_group.add_argument("--seed", type=int, default=42, help="Random seed for initialization.")
    training_args_group.add_argument(
        "--report_to", type=str, default="all", help="The list of integrations to report results and logs to."
    )

    # Weights & Biases arguments
    wandb_args = parser.add_argument_group("Weights & Biases Arguments")
    wandb_args.add_argument("--run_name", type=str, default=None, help="W&B run name.")
    wandb_args.add_argument("--project_name", type=str, default=None, help="W&B project name.")
    wandb_args.add_argument("--tags", type=str, nargs="*", default=None, help="W&B tags for the run.")

    training_args_group.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="The path to a checkpoint to resume training from.",
    )
    training_args_group.add_argument(
        "--load_best_model_at_end",
        type=str_to_bool,
        default=True,
        help=(
            "Whether to load the best model found during training at the end of training. "
            "This enables HuggingFace's built-in best metric tracking, which is used by "
            "top-K checkpointing. Recommended to keep as True."
        ),
    )

    # Custom Optimizer Arguments
    optimizer_args = parser.add_argument_group("Optimizer Arguments")
    optimizer_args.add_argument(
        "--optimizer_name",
        type=str,
        default="adamw",
        choices=["adamw", "stableadamw"],
        help=(
            "Name of the optimizer to use. Choose between 'adamw' (torch.optim.AdamW) or "
            "'stableadamw' (optimi.StableAdamW)."
        ),
    )

    # Custom LR Scheduler Arguments
    lr_scheduler_args = parser.add_argument_group("Custom Learning Rate Scheduler Arguments")
    lr_scheduler_args.add_argument(
        "--lr_scheduler_name",
        type=str,
        default=None,
        choices=["wsd"],
        help=(
            "Name of the custom learning rate scheduler to use. 'wsd' for Warmup-Stable-Decay. "
            "If not provided, uses HuggingFace's default scheduler as specified by --lr_scheduler_type. "
            "If 'wsd' is used, --lr_scheduler_type is ignored."
        ),
    )
    lr_scheduler_args.add_argument(
        "--wsd_num_decay_steps",
        type=int,
        default=None,
        help="Number of decay steps for WSD scheduler. Required if --lr_scheduler_name is 'wsd'.",
    )
    lr_scheduler_args.add_argument(
        "--wsd_decay_type",
        type=str,
        default="cosine",
        choices=["cosine", "linear"],
        help="Decay type for WSD scheduler.",
    )

    # Top-K Checkpointing Arguments
    checkpoint_args = parser.add_argument_group("Top-K Checkpointing Arguments")
    checkpoint_args.add_argument(
        "--save_top_k",
        type=int,
        default=0,
        help=(
            "Number of best checkpoints to keep based on validation metric. "
            "If <= 0, disables top-K checkpointing and uses standard HF checkpointing. "
            "Defaults to 0 (disabled)."
        ),
    )
    checkpoint_args.add_argument(
        "--metric_for_best_model",
        type=str,
        default="eval_loss",
        help=(
            "The metric to use for determining the best model for top-K checkpointing. "
            "Defaults to 'eval_loss'."
        ),
    )
    checkpoint_args.add_argument(
        "--greater_is_better",
        type=str_to_bool,
        default=False,
        help=(
            "Whether higher values of the metric indicate better models. "
            "Set to False for loss-based metrics, True for accuracy-based metrics. "
            "Defaults to False."
        ),
    )
    checkpoint_args.add_argument(
        "--save_best_checkpoint_only",
        type=str_to_bool,
        default=False,
        help=(
            "If True, only saves checkpoints that improve the best metric. "
            "If False, saves all checkpoints but only keeps top K. "
            "Defaults to False."
        ),
    )
    checkpoint_args.add_argument(
        "--skip_first_n_steps",
        type=int,
        default=1000,
        help=(
            "Skip top-K checkpointing for the first N steps to avoid saving too many "
            "early checkpoints during the noisy initial training phase. "
            "Defaults to 1000 steps."
        ),
    )

    # Variable Save Frequency (Optional)
    checkpoint_args.add_argument(
        "--early_save_until_step",
        type=int,
        default=0,
        help=("Apply higher save frequency until this global step. Set 0 to disable the early ramp."),
    )
    checkpoint_args.add_argument(
        "--early_save_every",
        type=int,
        default=None,
        help=("Force an additional save every N steps during the early phase (<= early_save_until_step)."),
    )
    checkpoint_args.add_argument(
        "--late_save_every",
        type=int,
        default=None,
        help=(
            "After the early phase, force an additional save every N steps. Leave None to rely on base save_steps."
        ),
    )

    # Early Stopping Arguments
    early_stopping_args = parser.add_argument_group("Early Stopping Arguments")
    early_stopping_args.add_argument(
        "--early_stopping_patience",
        type=int,
        default=0,
        help=(
            "Number of evaluations with no improvement after which training will be stopped. "
            "If <= 0, disables early stopping. Uses the same metric as specified in "
            "--metric_for_best_model. Defaults to 0 (disabled)."
        ),
    )
    early_stopping_args.add_argument(
        "--early_stopping_threshold",
        type=float,
        default=0.0,
        help=(
            "How much the specified metric must improve to satisfy early stopping conditions. "
            "For example, if monitoring 'eval_loss' with threshold=0.01, training will stop "
            "if the loss doesn't improve by at least 0.01. Defaults to 0.0."
        ),
    )

    logger = setup_logging()

    # Let argparse handle --help / -h before attempting config-file pre-parsing,
    # otherwise the temp_parser below would try to interpret "--help" as a path.
    if "-h" in sys.argv or "--help" in sys.argv:
        parser.parse_args()  # prints help and exits

    temp_parser = argparse.ArgumentParser(add_help=False)
    temp_parser.add_argument("config", type=Path)
    config_path_args, _ = temp_parser.parse_known_args()

    config_from_file = {}
    if config_path_args.config:  # If a config path was provided
        if config_path_args.config.exists():
            logger.info(f"Loading configuration from: {config_path_args.config}")
            config_from_file = read_json_yaml(config_path_args.config)
            if not isinstance(config_from_file, dict):
                logger.error(
                    f"Configuration file {config_path_args.config} must be a YAML/JSON mapping (dictionary)."
                )
                sys.exit(1)
            parser.set_defaults(**config_from_file)
        else:
            logger.error(f"Specified config file {config_path_args.config} not found. Please check the path.")
            sys.exit(1)
    else:
        logger.warning(
            "No config file path recognized by pre-parser. Proceeding without config file defaults."
        )

    args = parser.parse_args()

    # Validate required arguments before any side effects (e.g. W&B run creation).
    required_params = ["train_path", "val_path", "output_dir"]
    missing_params = [param for param in required_params if getattr(args, param) is None]

    if missing_params:
        parser.error(
            "The following arguments are required but were not provided via CLI or config "
            f"file: {', '.join(missing_params)}"
        )

    # Initialize Weights and Biases logging
    if args.report_to == "wandb" or args.report_to == "all" or "wandb" in args.report_to:
        wandb_init(args, logger)

    # Log all final parsed arguments
    logger.info(f"All parsed arguments: {args}")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare data
    train_dataset, train_sampler, val_dataset, val_sampler = prepare_data(
        args.train_path,
        args.val_path,
        args.max_tokens_per_batch,
        args.min_patients_per_batch,
        args.max_seq_length_per_patient,
        args.token_dropout_prob,
        args.min_patient_length,
        args.stride,
        args.num_ntp_classes,
        convert_ages_to_positions=args.convert_ages_to_positions,
    )

    # Safety: if data appears to be positions mode (e.g., demographic prefix), force-disable use_normed_ages
    if getattr(args, "use_normed_ages", True):
        if _detect_positions_mode(train_dataset, logger):
            logger.warning(
                "Detected ages as token positions and age_normalized as zeros; "
                "forcing use_normed_ages=False to avoid double-counting positions."
            )
            args.use_normed_ages = False

    # Prepare model
    model = prepare_model(args, device, logger)

    # Start training
    train_model(args, model, train_dataset, train_sampler, val_dataset, val_sampler, logger)


if __name__ == "__main__":
    main()
