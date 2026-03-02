import json
import math
import os
import shutil

import torch
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from ehr_fm.logger import setup_logging


class TopKCheckpointCallback(TrainerCallback):
    """
    A callback that implements top-K checkpointing based on a specified metric.

    Saves checkpoints immediately when a new top-K worthy result is found, independent
    of the regular checkpoint frequency.
    """

    def __init__(
        self,
        save_top_k: int = 3,
        metric_name: str = "eval_loss",
        greater_is_better: bool = False,
        delete_checkpoint_callback: bool = True,
        skip_first_n_steps: int = 0,
    ):
        self.save_top_k = save_top_k
        self.metric_name = metric_name
        self.greater_is_better = greater_is_better
        self.delete_checkpoint_callback = delete_checkpoint_callback
        self.skip_first_n_steps = skip_first_n_steps

        # Track best checkpoints: List[Tuple[metric_value, checkpoint_dir, step]]
        self.best_checkpoints: list[tuple[float, str, int]] = []
        self._saving_checkpoint = False
        self._trainer = None
        self.logger = setup_logging()

        self.logger.info(
            f"TopKCheckpointCallback: top_k={self.save_top_k}, metric={self.metric_name}, "
            f"greater_is_better={self.greater_is_better}, skip_first_n={self.skip_first_n_steps}"
        )

    def setup_trainer(self, trainer):
        self._trainer = trainer

    def _save_metadata(self, output_dir: str):
        metadata = {
            "save_top_k": self.save_top_k,
            "metric_name": self.metric_name,
            "greater_is_better": self.greater_is_better,
            "skip_first_n_steps": self.skip_first_n_steps,
            "best_checkpoints": [
                {"metric_value": val, "checkpoint_dir": path, "step": step}
                for val, path, step in self.best_checkpoints
            ],
        }

        metadata_path = os.path.join(output_dir, "top_k_checkpoints.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def _load_metadata(self, output_dir: str):
        metadata_path = os.path.join(output_dir, "top_k_checkpoints.json")
        if not os.path.exists(metadata_path):
            return

        try:
            with open(metadata_path) as f:
                metadata = json.load(f)

            self.best_checkpoints = [
                (item["metric_value"], item["checkpoint_dir"], item["step"])
                for item in metadata.get("best_checkpoints", [])
                if os.path.exists(item["checkpoint_dir"])
            ]
        except (json.JSONDecodeError, KeyError) as e:
            self.logger.warning(f"Could not load top-K checkpoint metadata: {e}")

    def _is_worthy_checkpoint(self, metric_value: float, current_step: int) -> bool:
        if current_step <= self.skip_first_n_steps:
            return False

        if len(self.best_checkpoints) < self.save_top_k:
            return True

        worst_metric = self.best_checkpoints[-1][0]  # Last item after sorting
        return (metric_value > worst_metric) if self.greater_is_better else (metric_value < worst_metric)

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if self.save_top_k > 0 and args.output_dir:
            self.logger.info(
                f"Initializing top-K checkpointing: save_top_k={self.save_top_k}, metric={self.metric_name}"
            )
            self._load_metadata(args.output_dir)

            if self.best_checkpoints:
                self.logger.info(f"Loaded {len(self.best_checkpoints)} existing top-K checkpoints")

    def on_evaluate(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs
    ):
        if self.save_top_k <= 0 or self._saving_checkpoint:
            return control

        # Get current metric value
        current_metric = None
        if logs and self.metric_name in logs:
            current_metric = logs[self.metric_name]
        elif state.log_history:
            for log_entry in reversed(state.log_history):
                if self.metric_name in log_entry:
                    current_metric = log_entry[self.metric_name]
                    break

        if current_metric is None:
            return control

        if math.isnan(current_metric) or math.isinf(current_metric):
            self.logger.warning(
                f"Rejecting checkpoint with invalid metric: {self.metric_name}={current_metric} "
                f"at step {state.global_step}"
            )
            return control

        # Save checkpoint if worthy
        if self._is_worthy_checkpoint(current_metric, state.global_step):
            self.logger.info(
                f"Saving top-K checkpoint: {self.metric_name}={current_metric:.6f} at step {state.global_step}"
            )
            self._save_topk_checkpoint(args, state, current_metric)

        return control

    def _save_topk_checkpoint(self, args: TrainingArguments, state: TrainerState, metric_value: float):
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-topk-{state.global_step}")
        self._saving_checkpoint = True

        try:
            if self._trainer is None:
                raise ValueError("Trainer reference not set. Call setup_trainer() when adding callback.")

            os.makedirs(checkpoint_dir, exist_ok=True)
            self._trainer.save_model(checkpoint_dir)
            state.save_to_json(os.path.join(checkpoint_dir, "trainer_state.json"))
            torch.save(args, os.path.join(checkpoint_dir, "training_args.bin"))

            if self._trainer.optimizer is not None:
                torch.save(self._trainer.optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pt"))

            if self._trainer.lr_scheduler is not None:
                torch.save(
                    self._trainer.lr_scheduler.state_dict(), os.path.join(checkpoint_dir, "scheduler.pt")
                )

            self.logger.info(f"Saved complete top-K checkpoint to: {checkpoint_dir}")
            self._update_best_checkpoints(metric_value, checkpoint_dir, state.global_step)

        except Exception as e:
            self.logger.error(f"Error saving top-K checkpoint: {e}")
            # Clean up partial checkpoint if it was created
            if os.path.exists(checkpoint_dir):
                try:
                    shutil.rmtree(checkpoint_dir)
                    self.logger.info(f"Cleaned up partial checkpoint: {checkpoint_dir}")
                except Exception as e:
                    self.logger.error(f"Failed to clean up partial checkpoint: {checkpoint_dir}")
                    raise e
        finally:
            self._saving_checkpoint = False

    def _update_best_checkpoints(self, metric_value: float, checkpoint_dir: str, step: int):
        self.logger.info(f"Adding checkpoint to top-K tracking: step={step}, metric={metric_value:.6f}")
        self.best_checkpoints.append((metric_value, checkpoint_dir, step))
        self.best_checkpoints.sort(key=lambda x: x[0], reverse=self.greater_is_better)

        self.logger.info(
            f"Before cleanup: {len(self.best_checkpoints)} checkpoints, save_top_k={self.save_top_k}"
        )

        removed_count = 0
        while len(self.best_checkpoints) > self.save_top_k:
            _, worst_checkpoint_dir, worst_step = self.best_checkpoints.pop()
            removed_count += 1

            self.logger.info(
                f"Removing checkpoint {removed_count}: {worst_checkpoint_dir} (step {worst_step})"
            )

            if self.delete_checkpoint_callback and os.path.exists(worst_checkpoint_dir):
                try:
                    shutil.rmtree(worst_checkpoint_dir)
                    self.logger.info(f"Successfully deleted checkpoint: {worst_checkpoint_dir}")
                except OSError as e:
                    self.logger.error(f"Failed to delete checkpoint {worst_checkpoint_dir}: {e}")
            elif not self.delete_checkpoint_callback:
                self.logger.info(f"Checkpoint deletion disabled, keeping: {worst_checkpoint_dir}")
            else:
                self.logger.warning(f"Checkpoint directory doesn't exist: {worst_checkpoint_dir}")

        if self.best_checkpoints:
            self._save_metadata(os.path.dirname(checkpoint_dir))

        self.logger.info(f"Current top-{len(self.best_checkpoints)} checkpoints:")
        for i, (val, path, ckpt_step) in enumerate(self.best_checkpoints):
            self.logger.info(
                f"  {i+1}. Step {ckpt_step}: {self.metric_name}={val:.6f} - {os.path.basename(path)}"
            )

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if self.save_top_k > 0 and not self._saving_checkpoint and self.best_checkpoints:
            self._save_metadata(args.output_dir)
        return control

    def get_best_checkpoint_path(self) -> str | None:
        return self.best_checkpoints[0][1] if self.best_checkpoints else None

    def get_best_metric_value(self) -> float | None:
        return self.best_checkpoints[0][0] if self.best_checkpoints else None


class VariableSaveFrequencyCallback(TrainerCallback):
    """
    Trigger additional saves with a higher frequency during early training.

    This does not change `TrainingArguments.save_strategy`/`save_steps`; it only
    forces `should_save=True` on selected steps, so it composes with the base
    Trainer behavior and with Top-K checkpointing.
    """

    def __init__(
        self,
        early_save_until_step: int = 0,
        early_save_every: int | None = None,
        late_save_every: int | None = None,
    ):
        self.early_save_until_step = int(early_save_until_step or 0)
        self.early_save_every = int(early_save_every) if early_save_every else None
        self.late_save_every = int(late_save_every) if late_save_every else None
        self.logger = setup_logging()

        self.logger.info(
            "VariableSaveFrequencyCallback initialized: "
            f"early_until={self.early_save_until_step}, "
            f"early_every={self.early_save_every}, "
            f"late_every={self.late_save_every}"
        )

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if self.early_save_every is None and self.late_save_every is None:
            return control

        step = state.global_step
        if step <= 0:
            return control

        interval = None
        if self.early_save_every is not None and step <= self.early_save_until_step:
            interval = self.early_save_every
        elif self.late_save_every is not None and self.late_save_every > 0:
            interval = self.late_save_every

        if interval and interval > 0 and (step % interval == 0):
            control.should_save = True
        return control
