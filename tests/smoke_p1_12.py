"""P1.12 smoke tests: ScoreDistributionCalibrator"""
import sys; sys.path.insert(0, '.')
import torch
from model.score_calibration import ScoreDistributionCalibrator
from model.score_utils import independent_poisson_score_grid

def test_identity():
    cal = ScoreDistributionCalibrator()
    base = independent_poisson_score_grid(1.5, 1.2)
    out = cal.calibrate(1.5, 1.2)
    diff = abs(base['home_win_prob'] - out['home_win_prob'])
    assert diff < 0.001, f"Identity diff {diff}"
    s = out['score_matrix'].sum().item()
    assert abs(s - 1.0) < 0.01, f"Grid sum {s}"
    print("PASS: identity")

def test_temperature_increases_entropy():
    base = independent_poisson_score_grid(1.5, 1.2)['score_matrix']
    cal = ScoreDistributionCalibrator(total_temperature=2.0)
    out = cal.calibrate(1.5, 1.2)['score_matrix']
    ent_base = -(base * torch.log(base + 1e-10)).sum().item()
    ent_out = -(out * torch.log(out + 1e-10)).sum().item()
    assert ent_out > ent_base, f"Entropy not increased: base={ent_base:.4f} warm={ent_out:.4f}"
    assert abs(out.sum().item() - 1.0) < 0.01
    print(f"PASS: temperature increases entropy ({ent_base:.4f} -> {ent_out:.4f})")

def test_lambda_scaling():
    cal = ScoreDistributionCalibrator(lambda_scale=1.2)
    base = independent_poisson_score_grid(1.5, 1.2)
    out = cal.calibrate(1.5, 1.2)
    assert out['expected_total_goals'] > base['expected_total_goals']
    print(f"PASS: lambda scaling ({base['expected_total_goals']} -> {out['expected_total_goals']})")

def test_dixon_coles():
    base = independent_poisson_score_grid(1.5, 1.2)['score_matrix']
    cal = ScoreDistributionCalibrator(dc_rho=0.3)
    out = cal.calibrate(1.5, 1.2)['score_matrix']
    # rho>0 should increase 0-0 and 1-1
    assert out[0,0].item() > base[0,0].item(), "DC should increase P(0-0)"
    assert out[1,1].item() > base[1,1].item(), "DC should increase P(1-1)"
    assert abs(out.sum().item() - 1.0) < 0.01
    print(f"PASS: DC correction (0-0: {base[0,0].item():.4f} -> {out[0,0].item():.4f})")

def test_no_nan_inf():
    configs = [
        ("warm", ScoreDistributionCalibrator(total_temperature=2.0)),
        ("cold", ScoreDistributionCalibrator(total_temperature=0.5)),
        ("scale", ScoreDistributionCalibrator(lambda_scale=0.5)),
        ("dc_pos", ScoreDistributionCalibrator(dc_rho=0.3)),
        ("dc_neg", ScoreDistributionCalibrator(dc_rho=-0.3)),
        ("tail", ScoreDistributionCalibrator(tail_boost=0.1)),
        ("all", ScoreDistributionCalibrator(total_temperature=1.5, diff_temperature=1.3, lambda_scale=1.1, dc_rho=0.1, tail_boost=0.05)),
    ]
    for lambdas in [(0.5, 3.0), (0.1, 0.1), (5.0, 5.0), (0.01, 0.01)]:
        for name, calib in configs:
            o = calib.calibrate(lambdas[0], lambdas[1])
            g = o['score_matrix']
            assert not torch.isnan(g).any(), f"{name} @ {lambdas} has NaN"
            assert not torch.isinf(g).any(), f"{name} @ {lambdas} has Inf"
            assert g.min() >= 0, f"{name} @ {lambdas} has negative probs"
            assert abs(g.sum().item() - 1.0) < 0.02, f"{name} @ {lambdas} sum={g.sum().item()}"
    print("PASS: no NaN/Inf, all non-negative, all sum~1")

def test_save_load():
    cal = ScoreDistributionCalibrator(total_temperature=1.5, dc_rho=0.1)
    cal.save("data/reports/p1_12/_test_config.json")
    cal2 = ScoreDistributionCalibrator.load("data/reports/p1_12/_test_config.json")
    assert cal2.total_temperature == 1.5
    assert cal2.dc_rho == 0.1
    print("PASS: save/load")


def test_disagreement_policy():
    from model.score_utils import disagreement_policy
    # High agreement
    r = disagreement_policy(
        {"home": 0.5, "draw": 0.3, "away": 0.2},
        {"home_win_prob": 0.48, "draw_prob": 0.30, "away_win_prob": 0.22},
        0.05, True
    )
    assert r['agreement_level'] == 'high_agreement'
    assert r['confidence_adjustment'] == 'keep'

    # Same winner, high JS
    r2 = disagreement_policy(
        {"home": 0.4, "draw": 0.3, "away": 0.3},
        {"home_win_prob": 0.6, "draw_prob": 0.2, "away_win_prob": 0.2},
        0.15, True
    )
    assert r2['agreement_level'] == 'same_winner_high_js'
    assert r2['confidence_adjustment'] == 'soften'

    # Winner disagreement
    r3 = disagreement_policy(
        {"home": 0.4, "draw": 0.3, "away": 0.3},
        {"home_win_prob": 0.2, "draw_prob": 0.2, "away_win_prob": 0.6},
        0.15, False
    )
    assert r3['agreement_level'] == 'winner_disagreement'
    assert r3['confidence_adjustment'] == 'flag_review'

    # Score extreme, Euro conservative
    r4 = disagreement_policy(
        {"home": 0.4, "draw": 0.3, "away": 0.3},
        {"home_win_prob": 0.8, "draw_prob": 0.1, "away_win_prob": 0.1},
        0.08, True
    )
    assert r4['agreement_level'] == 'score_extreme_euro_conservative'
    print("PASS: disagreement_policy all 4 levels")


def test_inference_with_calibration():
    import json
    # Quick load model and run predict with calibration config
    import torch
    from model.model_oddsmind import OddsMindConfig, OddsMindModel
    from infer.predict_odds_match import predict
    cfg = OddsMindConfig(hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                         asian_num_classes=3, score_head_version='v2')
    model = OddsMindModel(cfg)
    model.eval()
    # Create minimal match dict
    match = {
        "odds_timeline": [
            {"euro_h": 2.0, "euro_d": 3.0, "euro_a": 4.0, "asian_line": -1.0,
             "upper_water": 0.9, "lower_water": 0.9, "minutes_before_kickoff": 90}
        ] * 5,
        "cutoff_minutes": 0,
    }
    # Save dummy calibration config
    cal = ScoreDistributionCalibrator(total_temperature=1.5)
    cal.save("data/reports/p1_12/_test_infer_config.json")

    r = predict(model, match, 'cpu', score_loss_type='poisson',
                calibration_config="data/reports/p1_12/_test_infer_config.json")
    assert 'score_calibrated_1x2' in r, "Missing calibrated 1X2"
    assert 'calibrated_top_k_scorelines' in r, "Missing calibrated top-k"
    assert 'disagreement_policy' in r, "Missing disagreement_policy"
    assert r['disagreement_policy']['agreement_level'] in [
        'high_agreement', 'same_winner_high_js', 'winner_disagreement',
        'score_extreme_euro_conservative'
    ]
    print("PASS: inference with calibration outputs all fields")


def test_no_tokenizer_vocab_lmhead():
    """Verify no tokenizer/vocab/LM Head imports in P1.12 code."""
    import os, sys
    forbidden = ['tokenizer', 'vocab', 'lm_head', 'LMHead', 'next_token']
    files_to_check = [
        'model/score_calibration.py',
        'model/score_utils.py',
        'eval/eval_p1_12_score_calibration.py',
    ]
    for fpath in files_to_check:
        with open(fpath, encoding='utf-8') as f:
            content = f.read().lower()
        for word in forbidden:
            assert word not in content, f"{fpath} contains forbidden word: {word}"
    print("PASS: no tokenizer/vocab/LM Head in P1.12 code")


if __name__ == "__main__":
    test_identity()
    test_temperature_increases_entropy()
    test_lambda_scaling()
    test_dixon_coles()
    test_no_nan_inf()
    test_save_load()
    test_disagreement_policy()
    test_inference_with_calibration()
    test_no_tokenizer_vocab_lmhead()
    print("\nALL P1.12 SMOKE TESTS PASSED")
