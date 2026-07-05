# P8 Frozen Draw Signal Forensics Report

## Verdict

- recommended_next_phase: `P9 Market-Prior Anchored Residual Model`
- P8_P6_SUPPRESSES_DRAW_BELOW_MARKET
- P8_STOP_DRAW_ARGMAX_PRIMARY_GATE

## Protocol

- hard_fail_reasons: `[]`
- test split: `not loaded`
- official val fit: `false`
- json: `runs/p8_draw_signal_forensics/p8_final_report.json`
- diagnostics: `diagnostics/p8_draw_signal_forensics/`

## Label / Split Audit

- train_samples: `26441`
- val_samples: `4747`
- train_draw_rate: `0.19136946408986044`
- val_draw_rate: `0.1938066147040236`
- train_home_rate: `0.4777807193373927`
- val_home_rate: `0.4687170844744049`
- train_away_rate: `0.33084981657274687`
- val_away_rate: `0.33747630082157154`
- class_mapping_integrity: `True`
- duplicate_match_ids: `[]`
- invalid_odds_values: `78`
- odds_after_kickoff_count: `0`
- latest_event_after_kickoff_count: `0`
- hard_fail_reasons: `[]`

## Market Close vs P6

- market_logloss: `0.9431851506233215`
- market_ece: `0.041227`
- market_draw_nll: `1.3941060304641724`
- market_draw_recall_argmax: `0.0010869565217391304`
- market_draw_precision_argmax: `0.25`
- market_draw_top2: `0.6315217614173889`
- market_mean_p_draw_true_draw: `0.25324422121047974`
- market_mean_draw_margin_to_top: `0.2937964200973511`
- market_draw_rank1_count: `4`
- market_draw_rank2_count: `3227`
- market_draw_rank3_count: `1516`
- p6_val_logloss: `0.9378808736801147`
- p6_ece: `0.03557`
- p6_classwise_nll_draw: `1.6407524347305298`
- p6_draw_recall_argmax: `0.0010869565217391304`
- p6_draw_top2: `0.33043476939201355`
- p6_mean_p_draw_true_draw: `0.19863468408584595`
- p6_mean_draw_margin_to_top: `0.399319589138031`
- mean_p6_minus_market_p_draw_true_draw: `-0.05460951430779673`
- p6_draw_shrinkage_ratio: `0.784360278034921`
- p6_vs_market_draw_corr: `0.9821014504423421`
- p6_vs_market_draw_mae: `0.05271845913926732`

## Probe Summary

| Probe | Seeds | Logloss | Draw NLL | Draw Top2 | Mean P(draw true draw) | Draw Recall |
|---|---:|---:|---:|---:|---:|---:|
| probe_0_train_class_prior | 1 | 1.0399442911148071 | 1.6535494327545166 | 0.0 | 0.19136947393417358 | 0.0 |
| probe_1_market_close_de_vig | 1 | 0.9431851506233215 | 1.3941060304641724 | 0.6315217614173889 | 0.25324422121047974 | 0.0010869565217391304 |
| probe_2_market_open_de_vig | 1 | 0.9522606730461121 | 1.415837049484253 | 0.585869550704956 | 0.24777500331401825 | 0.0010869565217391304 |
| probe_3_train_only_multinomial_logistic_market_features | 3 | 0.939432164033254 | 1.6753637393315632 | 0.3347826103369395 | 0.1923668384552002 | 0.0010869565217391304 |
| probe_4_train_only_binary_draw_then_conditional_home_away_logistic | 3 | 0.9395357370376587 | 1.5923430919647217 | 0.416304349899292 | 0.20812671879927316 | 0.0 |
| probe_5_train_only_tiny_tabular_mlp_market_features | 3 | 0.9430789152781168 | 1.6677343448003132 | 0.3177536229292552 | 0.19495290517807007 | 0.0 |

## Interpretation

- Market close is worse than P6 on overall logloss, but much better on draw-specific ranking and draw probability for true draws.
- P6 strongly shrinks draw probability relative to the market prior on true draws.
- Data/split hard-stop checks pass, so the next useful model phase is market-prior anchored residual modeling, not a larger backbone.
