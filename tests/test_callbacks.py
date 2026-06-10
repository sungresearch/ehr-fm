"""Tests for ehr_fm.callbacks -- VariableSaveFrequencyCallback."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from ehr_fm.callbacks import VariableSaveFrequencyCallback

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_args(output_dir: str) -> SimpleNamespace:
    return SimpleNamespace(output_dir=output_dir)


def _make_state(global_step: int = 100, log_history: list | None = None) -> SimpleNamespace:
    state = SimpleNamespace(global_step=global_step, log_history=log_history or [])
    state.save_to_json = MagicMock()
    return state


def _make_control() -> SimpleNamespace:
    return SimpleNamespace(should_save=False)


# ===================================================================
# VariableSaveFrequencyCallback
# ===================================================================


class TestVariableSaveFrequencyInit:
    def test_defaults(self):
        cb = VariableSaveFrequencyCallback()
        assert cb.early_save_until_step == 0
        assert cb.early_save_every is None
        assert cb.late_save_every is None

    def test_custom_params(self):
        cb = VariableSaveFrequencyCallback(
            early_save_until_step=1000,
            early_save_every=100,
            late_save_every=500,
        )
        assert cb.early_save_until_step == 1000
        assert cb.early_save_every == 100
        assert cb.late_save_every == 500


class TestVariableSaveOnStepEnd:
    def test_both_none_noop(self):
        cb = VariableSaveFrequencyCallback()
        control = _make_control()
        cb.on_step_end(_make_args("/tmp"), _make_state(global_step=100), control)
        assert control.should_save is False

    def test_step_zero_noop(self):
        cb = VariableSaveFrequencyCallback(early_save_every=10)
        control = _make_control()
        cb.on_step_end(_make_args("/tmp"), _make_state(global_step=0), control)
        assert control.should_save is False

    def test_early_phase_matching_step(self):
        cb = VariableSaveFrequencyCallback(early_save_until_step=100, early_save_every=10)
        control = _make_control()
        cb.on_step_end(_make_args("/tmp"), _make_state(global_step=50), control)
        assert control.should_save is True

    def test_early_phase_non_matching_step(self):
        cb = VariableSaveFrequencyCallback(early_save_until_step=100, early_save_every=10)
        control = _make_control()
        cb.on_step_end(_make_args("/tmp"), _make_state(global_step=53), control)
        assert control.should_save is False

    def test_late_phase(self):
        cb = VariableSaveFrequencyCallback(
            early_save_until_step=100,
            early_save_every=10,
            late_save_every=50,
        )
        control = _make_control()
        cb.on_step_end(_make_args("/tmp"), _make_state(global_step=200), control)
        assert control.should_save is True

    def test_late_phase_non_matching(self):
        cb = VariableSaveFrequencyCallback(
            early_save_until_step=100,
            early_save_every=10,
            late_save_every=50,
        )
        control = _make_control()
        cb.on_step_end(_make_args("/tmp"), _make_state(global_step=201), control)
        assert control.should_save is False

    def test_only_late_save(self):
        cb = VariableSaveFrequencyCallback(late_save_every=25)
        control = _make_control()
        cb.on_step_end(_make_args("/tmp"), _make_state(global_step=75), control)
        assert control.should_save is True
