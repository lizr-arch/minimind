import json
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.p7_run_matrix import (
    FIXED_P7_SEEDS,
    FIXED_P7_VARIANTS,
    build_protocol_audit,
    parse_registered_seeds,
    parse_registered_variants,
    write_protocol_audit,
)


def test_p7_matrix_has_exact_registered_variants():
    assert list(FIXED_P7_VARIANTS) == [
        "p7_a_factorized_ce_only",
        "p7_b_factorized_ce_draw_bce_005",
        "p7_c_factorized_ce_draw_bce_010",
        "p7_d_factorized_ce_cond_ha_aux_005",
    ]


def test_p7_matrix_has_exact_registered_seeds():
    assert FIXED_P7_SEEDS == [42, 123, 2025]


def test_p7_runner_rejects_unregistered_variant():
    with pytest.raises(ValueError, match="Unregistered P7 variant"):
        parse_registered_variants("p7_a_factorized_ce_only,p7_x_extra")


def test_p7_runner_rejects_unregistered_seed():
    with pytest.raises(ValueError, match="Unregistered P7 seed"):
        parse_registered_seeds("42,7")


def test_p7_runner_rejects_test_split(tmp_path):
    audit = build_protocol_audit(
        baseline_path="runs/p6/euro_default",
        data_path="data.jsonl",
        train_ids="train_match_ids.txt",
        val_ids="test_match_ids.txt",
        variants=list(FIXED_P7_VARIANTS),
        seeds=FIXED_P7_SEEDS,
    )

    assert audit["no_test_split_loaded"] is False
    assert audit["hard_fail_reasons"] == ["P7_FAIL_TEST_SPLIT_REFERENCE"]


def test_p7_runner_writes_protocol_audit(tmp_path):
    audit = build_protocol_audit(
        baseline_path="runs/p6/euro_default",
        data_path="data.jsonl",
        train_ids="train_match_ids.txt",
        val_ids="val_match_ids.txt",
        variants=list(FIXED_P7_VARIANTS),
        seeds=FIXED_P7_SEEDS,
    )
    out = tmp_path / "protocol_audit.json"

    write_protocol_audit(audit, out)

    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["no_test_split_loaded"] is True
    assert loaded["candidate_count"] == 4
    assert loaded["candidate_count_mutation"] is False
    assert loaded["official_val_fixed_matrix_only"] is True
