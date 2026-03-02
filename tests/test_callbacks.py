"""Tests for ehr_fm.callbacks -- TopKCheckpointCallback and VariableSaveFrequencyCallback."""

import os
from types import SimpleNamespace
from unittest.mock import MagicMock

from ehr_fm.callbacks import TopKCheckpointCallback, VariableSaveFrequencyCallback

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
# TopKCheckpointCallback
# ===================================================================


class TestTopKCheckpointInit:
    def test_defaults(self):
        cb = TopKCheckpointCallback()
        assert cb.save_top_k == 3
        assert cb.metric_name == "eval_loss"
        assert cb.greater_is_better is False
        assert cb.delete_checkpoint_callback is True
        assert cb.skip_first_n_steps == 0
        assert cb.best_checkpoints == []

    def test_custom_params(self):
        cb = TopKCheckpointCallback(
            save_top_k=5,
            metric_name="eval_accuracy",
            greater_is_better=True,
            delete_checkpoint_callback=False,
            skip_first_n_steps=100,
        )
        assert cb.save_top_k == 5
        assert cb.metric_name == "eval_accuracy"
        assert cb.greater_is_better is True
        assert cb.delete_checkpoint_callback is False
        assert cb.skip_first_n_steps == 100

    def test_setup_trainer(self):
        cb = TopKCheckpointCallback()
        mock_trainer = MagicMock()
        cb.setup_trainer(mock_trainer)
        assert cb._trainer is mock_trainer


class TestIsWorthyCheckpoint:
    def test_skipped_before_threshold(self):
        cb = TopKCheckpointCallback(skip_first_n_steps=50)
        assert cb._is_worthy_checkpoint(0.5, current_step=30) is False

    def test_at_threshold_still_skipped(self):
        cb = TopKCheckpointCallback(skip_first_n_steps=50)
        assert cb._is_worthy_checkpoint(0.5, current_step=50) is False

    def test_room_in_list(self):
        cb = TopKCheckpointCallback(save_top_k=3)
        assert cb._is_worthy_checkpoint(0.5, current_step=100) is True

    def test_loss_better_than_worst(self):
        cb = TopKCheckpointCallback(save_top_k=2, greater_is_better=False)
        cb.best_checkpoints = [(0.3, "/a", 10), (0.5, "/b", 20)]
        assert cb._is_worthy_checkpoint(0.4, current_step=100) is True

    def test_loss_not_better_than_worst(self):
        cb = TopKCheckpointCallback(save_top_k=2, greater_is_better=False)
        cb.best_checkpoints = [(0.3, "/a", 10), (0.5, "/b", 20)]
        assert cb._is_worthy_checkpoint(0.6, current_step=100) is False

    def test_accuracy_better_than_worst(self):
        cb = TopKCheckpointCallback(save_top_k=2, greater_is_better=True)
        cb.best_checkpoints = [(0.9, "/a", 10), (0.7, "/b", 20)]
        assert cb._is_worthy_checkpoint(0.8, current_step=100) is True

    def test_accuracy_not_better_than_worst(self):
        cb = TopKCheckpointCallback(save_top_k=2, greater_is_better=True)
        cb.best_checkpoints = [(0.9, "/a", 10), (0.7, "/b", 20)]
        assert cb._is_worthy_checkpoint(0.6, current_step=100) is False


class TestMetadataPersistence:
    def test_roundtrip(self, tmp_path):
        cb = TopKCheckpointCallback(save_top_k=3)
        ckpt_dir = str(tmp_path / "ckpt1")
        os.makedirs(ckpt_dir)
        cb.best_checkpoints = [(0.3, ckpt_dir, 100)]

        cb._save_metadata(str(tmp_path))

        cb2 = TopKCheckpointCallback(save_top_k=3)
        cb2._load_metadata(str(tmp_path))
        assert len(cb2.best_checkpoints) == 1
        assert cb2.best_checkpoints[0] == (0.3, ckpt_dir, 100)

    def test_missing_file_is_noop(self, tmp_path):
        cb = TopKCheckpointCallback()
        cb._load_metadata(str(tmp_path))
        assert cb.best_checkpoints == []

    def test_corrupt_json(self, tmp_path):
        meta_path = tmp_path / "top_k_checkpoints.json"
        meta_path.write_text("{bad json")
        cb = TopKCheckpointCallback()
        cb._load_metadata(str(tmp_path))
        assert cb.best_checkpoints == []

    def test_stale_dirs_filtered(self, tmp_path):
        cb = TopKCheckpointCallback()
        existing_dir = str(tmp_path / "exists")
        os.makedirs(existing_dir)
        cb.best_checkpoints = [
            (0.3, existing_dir, 10),
            (0.5, str(tmp_path / "gone"), 20),
        ]
        cb._save_metadata(str(tmp_path))

        cb2 = TopKCheckpointCallback()
        cb2._load_metadata(str(tmp_path))
        assert len(cb2.best_checkpoints) == 1
        assert cb2.best_checkpoints[0][1] == existing_dir


class TestUpdateBestCheckpoints:
    def test_sorts_ascending_for_loss(self, tmp_path):
        cb = TopKCheckpointCallback(save_top_k=3, greater_is_better=False, delete_checkpoint_callback=False)
        d1, d2, d3 = str(tmp_path / "a"), str(tmp_path / "b"), str(tmp_path / "c")
        for d in (d1, d2, d3):
            os.makedirs(d)

        cb._update_best_checkpoints(0.5, d1, 10)
        cb._update_best_checkpoints(0.3, d2, 20)
        cb._update_best_checkpoints(0.4, d3, 30)

        metrics = [v for v, _, _ in cb.best_checkpoints]
        assert metrics == sorted(metrics)

    def test_sorts_descending_for_accuracy(self, tmp_path):
        cb = TopKCheckpointCallback(save_top_k=3, greater_is_better=True, delete_checkpoint_callback=False)
        d1, d2, d3 = str(tmp_path / "a"), str(tmp_path / "b"), str(tmp_path / "c")
        for d in (d1, d2, d3):
            os.makedirs(d)

        cb._update_best_checkpoints(0.7, d1, 10)
        cb._update_best_checkpoints(0.9, d2, 20)
        cb._update_best_checkpoints(0.8, d3, 30)

        metrics = [v for v, _, _ in cb.best_checkpoints]
        assert metrics == sorted(metrics, reverse=True)

    def test_excess_deleted(self, tmp_path):
        cb = TopKCheckpointCallback(save_top_k=2, greater_is_better=False, delete_checkpoint_callback=True)
        dirs = []
        for i in range(3):
            d = str(tmp_path / f"ckpt_{i}")
            os.makedirs(d)
            dirs.append(d)

        cb._update_best_checkpoints(0.5, dirs[0], 10)
        cb._update_best_checkpoints(0.3, dirs[1], 20)
        cb._update_best_checkpoints(0.4, dirs[2], 30)

        assert len(cb.best_checkpoints) == 2
        assert not os.path.exists(dirs[0])

    def test_excess_kept_when_disabled(self, tmp_path):
        cb = TopKCheckpointCallback(save_top_k=1, greater_is_better=False, delete_checkpoint_callback=False)
        d1, d2 = str(tmp_path / "a"), str(tmp_path / "b")
        for d in (d1, d2):
            os.makedirs(d)

        cb._update_best_checkpoints(0.5, d1, 10)
        cb._update_best_checkpoints(0.3, d2, 20)

        assert len(cb.best_checkpoints) == 1
        assert os.path.exists(d1)

    def test_metadata_written(self, tmp_path):
        cb = TopKCheckpointCallback(save_top_k=3, delete_checkpoint_callback=False)
        d = str(tmp_path / "sub" / "ckpt")
        os.makedirs(d)
        cb._update_best_checkpoints(0.5, d, 10)
        assert (tmp_path / "sub" / "top_k_checkpoints.json").exists()


class TestOnEvaluate:
    def test_disabled_when_top_k_zero(self, tmp_path):
        cb = TopKCheckpointCallback(save_top_k=0)
        result = cb.on_evaluate(
            _make_args(str(tmp_path)), _make_state(), _make_control(), logs={"eval_loss": 0.5}
        )
        assert result is not None

    def test_reentrant_guard(self, tmp_path):
        cb = TopKCheckpointCallback()
        cb._saving_checkpoint = True
        result = cb.on_evaluate(
            _make_args(str(tmp_path)), _make_state(), _make_control(), logs={"eval_loss": 0.5}
        )
        assert result is not None

    def test_metric_from_logs(self, tmp_path):
        cb = TopKCheckpointCallback(save_top_k=1)
        cb._save_topk_checkpoint = MagicMock()
        state = _make_state(global_step=100)
        cb.on_evaluate(_make_args(str(tmp_path)), state, _make_control(), logs={"eval_loss": 0.5})
        cb._save_topk_checkpoint.assert_called_once()

    def test_metric_from_log_history(self, tmp_path):
        cb = TopKCheckpointCallback(save_top_k=1)
        cb._save_topk_checkpoint = MagicMock()
        state = _make_state(global_step=100, log_history=[{"eval_loss": 0.5}])
        cb.on_evaluate(_make_args(str(tmp_path)), state, _make_control(), logs=None)
        cb._save_topk_checkpoint.assert_called_once()

    def test_metric_not_found(self, tmp_path):
        cb = TopKCheckpointCallback(save_top_k=1)
        cb._save_topk_checkpoint = MagicMock()
        state = _make_state(global_step=100, log_history=[])
        cb.on_evaluate(_make_args(str(tmp_path)), state, _make_control(), logs={})
        cb._save_topk_checkpoint.assert_not_called()

    def test_nan_rejected(self, tmp_path):
        cb = TopKCheckpointCallback(save_top_k=1)
        cb._save_topk_checkpoint = MagicMock()
        cb.on_evaluate(
            _make_args(str(tmp_path)), _make_state(), _make_control(), logs={"eval_loss": float("nan")}
        )
        cb._save_topk_checkpoint.assert_not_called()

    def test_inf_rejected(self, tmp_path):
        cb = TopKCheckpointCallback(save_top_k=1)
        cb._save_topk_checkpoint = MagicMock()
        cb.on_evaluate(
            _make_args(str(tmp_path)), _make_state(), _make_control(), logs={"eval_loss": float("inf")}
        )
        cb._save_topk_checkpoint.assert_not_called()


class TestOnSave:
    def test_writes_metadata(self, tmp_path):
        cb = TopKCheckpointCallback(save_top_k=1)
        ckpt = str(tmp_path / "ckpt")
        os.makedirs(ckpt)
        cb.best_checkpoints = [(0.3, ckpt, 10)]
        cb.on_save(_make_args(str(tmp_path)), _make_state(), _make_control())
        assert (tmp_path / "top_k_checkpoints.json").exists()

    def test_skips_when_empty(self, tmp_path):
        cb = TopKCheckpointCallback(save_top_k=1)
        cb.on_save(_make_args(str(tmp_path)), _make_state(), _make_control())
        assert not (tmp_path / "top_k_checkpoints.json").exists()


class TestOnTrainBegin:
    def test_loads_metadata(self, tmp_path):
        cb1 = TopKCheckpointCallback(save_top_k=2)
        ckpt = str(tmp_path / "ckpt")
        os.makedirs(ckpt)
        cb1.best_checkpoints = [(0.3, ckpt, 10)]
        cb1._save_metadata(str(tmp_path))

        cb2 = TopKCheckpointCallback(save_top_k=2)
        cb2.on_train_begin(_make_args(str(tmp_path)), _make_state(), _make_control())
        assert len(cb2.best_checkpoints) == 1


class TestGetters:
    def test_best_path_empty(self):
        cb = TopKCheckpointCallback()
        assert cb.get_best_checkpoint_path() is None

    def test_best_path(self):
        cb = TopKCheckpointCallback()
        cb.best_checkpoints = [(0.3, "/best", 10), (0.5, "/worst", 20)]
        assert cb.get_best_checkpoint_path() == "/best"

    def test_best_metric_empty(self):
        cb = TopKCheckpointCallback()
        assert cb.get_best_metric_value() is None

    def test_best_metric(self):
        cb = TopKCheckpointCallback()
        cb.best_checkpoints = [(0.3, "/best", 10), (0.5, "/worst", 20)]
        assert cb.get_best_metric_value() == 0.3


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
