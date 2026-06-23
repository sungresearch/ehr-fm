from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from ehr_fm.logger import setup_logging


class VariableSaveFrequencyCallback(TrainerCallback):
    """
    Trigger additional saves with a higher frequency during early training.

    This does not change `TrainingArguments.save_strategy`/`save_steps`; it only
    forces `should_save=True` on selected steps, so it composes with the base
    Trainer behavior.
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
