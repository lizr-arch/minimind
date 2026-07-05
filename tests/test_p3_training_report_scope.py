"""Regression tests for P3 training report baseline scope."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.p3_train_patch_itransformer import (
    _pass_fail_suffix as p3_pass_fail_suffix,
    _validation_match_ids_for_baseline as p3_validation_match_ids,
)
from tools.p3b_train_patch_itransformer import (
    _pass_fail_suffix as p3b_pass_fail_suffix,
    _validation_match_ids_for_baseline as p3b_validation_match_ids,
)


def test_baseline_match_ids_follow_sample_limiter_for_p3_scripts():
    samples = [
        {"match_id": "m2"},
        {"match_id": "m1"},
        {"match_id": "m2"},
    ]
    full_val_ids = {"m1", "m2", "m3", "m4"}

    assert p3_validation_match_ids(samples, full_val_ids) == ["m2", "m1"]
    assert p3b_validation_match_ids(samples, full_val_ids) == ["m2", "m1"]


def test_summary_status_suffix_is_ascii_for_windows_redirects():
    assert p3_pass_fail_suffix(True).isascii()
    assert p3_pass_fail_suffix(False).isascii()
    assert p3b_pass_fail_suffix(True).isascii()
    assert p3b_pass_fail_suffix(False).isascii()
