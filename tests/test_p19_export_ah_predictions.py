import json
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.p19_export_ah_predictions import build_export_report_payload, prepare_export_output_dir


def test_p19_export_report_payload_records_no_training_policy():
    payload = build_export_report_payload(
        split_name="forward",
        checkpoint_dir="runs/p19/fold_01/seed_42/cover",
        ids_path="runs/p19/folds/fold_01/forward_match_ids.txt",
        row_count=123,
        metrics={"logloss": 0.93, "ah_cover_acc": 0.44},
    )

    assert payload["phase"] == "P19"
    assert payload["split_name"] == "forward"
    assert payload["input_policy"]["no_training"] is True
    assert payload["input_policy"]["no_test_split_loaded"] is True
    assert payload["metrics"]["logloss"] == pytest.approx(0.93)
    json.dumps(payload)


def test_p19_export_refuses_existing_prediction_outputs(tmp_path):
    (tmp_path / "val_ah_cover_predictions.csv").write_text("x\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        prepare_export_output_dir(tmp_path, allow_overwrite=False)

    prepare_export_output_dir(tmp_path, allow_overwrite=True)
