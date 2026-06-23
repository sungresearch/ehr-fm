"""Tests for ehr_fm.cli_utils -- str_to_bool and normalize_resume_checkpoint."""

import argparse

import pytest

from ehr_fm.cli_utils import normalize_resume_checkpoint, str_to_bool


class TestStrToBool:
    def test_bool_passthrough(self):
        assert str_to_bool(True) is True
        assert str_to_bool(False) is False

    @pytest.mark.parametrize("value", ["true", "True", "TRUE", "1", "yes", "  Yes  "])
    def test_truthy_strings(self, value):
        assert str_to_bool(value) is True

    @pytest.mark.parametrize("value", ["false", "False", "0", "no", " NO "])
    def test_falsy_strings(self, value):
        assert str_to_bool(value) is False

    @pytest.mark.parametrize("value", ["maybe", "", "2", "t", "y"])
    def test_invalid_raises(self, value):
        with pytest.raises(argparse.ArgumentTypeError):
            str_to_bool(value)


class TestNormalizeResumeCheckpoint:
    def test_none_is_none(self):
        assert normalize_resume_checkpoint(None) is None

    def test_bool_true_means_autodetect(self):
        assert normalize_resume_checkpoint(True) is True

    def test_bool_false_means_none(self):
        assert normalize_resume_checkpoint(False) is None

    @pytest.mark.parametrize("value", ["true", "1", "yes"])
    def test_truthy_string_means_autodetect(self, value):
        assert normalize_resume_checkpoint(value) is True

    @pytest.mark.parametrize("value", ["false", "0", "no", "none", "null", ""])
    def test_falsy_string_means_none(self, value):
        assert normalize_resume_checkpoint(value) is None

    def test_path_string_is_preserved(self):
        assert normalize_resume_checkpoint("/ckpt/checkpoint-5000") == "/ckpt/checkpoint-5000"
