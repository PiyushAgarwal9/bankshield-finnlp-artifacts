# Fairness evaluation release (output-mode the fairness table (Table 5))

Reproduces BankShield's Table 5 fairness metrics with no GPU or model download.

    python3 reproduce_table6.py
    # -> catch 66.7 / over-flag 4.9 / flip 0.0  -> REPRODUCED

## Contents
- `catch_action_detail.json`   per-item output-side action (BLOCK/ROUTE/ALLOW) + text SHA-256 for the 123 biased items (deployed service, output mode, thr 0.70); reproduce_table6.py audits it: 82 block / 0 route, block-only catch = 66.7%.
- `policy_config.json`          disclosable decision configuration (direction, 0.70 threshold, decision rule).
- `catch_ours_out.json`        123 biased items with output-mode verdicts (catch = 82/123).
- `release_benign_output.json` 185 benign identity-mention items, output-mode decisions (over-flag = 9/185).
- `release_minpairs_output.json` 560 minimal-pair items / 160 sets, output-mode decisions (flip = 0/160).
- `expected_metrics.json`      the reported values + Wilson CIs + McNemar p-values.
- `ENVIRONMENT.txt`            library and model versions.
- `reproduce_table6.py`        recomputes all three metrics from the logs above.

## Scope
These artifacts make the reported metrics reproducible from the released per-item predictions
(run `reproduce_table6.py`). They do NOT include a probe checkpoint: end-to-end regeneration of
BankShield predictions additionally requires the proprietary LoRA adapter weights and production
policy rules, which are withheld under a commercial/security exception granted in writing by the
workshop organizers, and the exact deployed discrimination-probe checkpoint, which is unavailable.
No probe reconstruction is shipped. The base model (google/gemma-4-12B-it) is public.
