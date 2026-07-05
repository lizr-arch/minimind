import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.p18_ah_threshold_sweep import direction_stats, summarize_seed_pairs


def test_p18_threshold_sweep_filters_by_abs_expected_units():
    pairs = [(0.20, 1.0, 0.90, 0.80), (-0.30, -1.0, 0.90, 0.80), (0.01, -1.0, 0.90, 0.80), (-0.02, 1.0, 0.90, 0.80)]

    summary = direction_stats(pairs, threshold=0.10, total_n=4)

    assert summary["n"] == 2
    assert summary["coverage"] == pytest.approx(0.5)
    assert summary["model_side_avg_units"] == pytest.approx(1.0)
    assert summary["model_side_avg_payoff"] == pytest.approx(0.85)
    assert summary["direction_accuracy_ex_push"] == pytest.approx(1.0)


def test_p18_threshold_sweep_aggregates_across_seeds():
    rows = summarize_seed_pairs(
        "toy",
        {
            42: [(0.20, 1.0), (0.01, -1.0)],
            123: [(-0.20, -1.0), (0.01, 1.0)],
        },
        [0.0, 0.10],
    )

    assert rows[0]["model"] == "toy"
    assert rows[1]["threshold"] == pytest.approx(0.10)
    assert rows[1]["mean_n"] == pytest.approx(1.0)
    assert rows[1]["mean_model_side_avg_units"] == pytest.approx(1.0)
