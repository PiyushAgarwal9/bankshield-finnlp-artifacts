# BankShield — reproducibility package

Per-item prediction logs and analyzers behind the paper's tables.

## Layout
- `paper/` — camera-ready paper (PDF) and LaTeX source.
- `data/` — frozen benchmark sets and prompt sheets.
- `results/` — per-item prediction logs, `<system>__<benchmark>__<direction>.json`.
- `harness/` — analysis scripts (regenerate tables from released logs; adapters under `harness/wrappers/` are reference code).
- `external_panel/` — external-benchmark competence panel + decontamination reproduction.
- `annotation/` — inter-annotator-agreement package.

## Reproduce tables (no GPU, no model)
    python3 results/table6_bharatbbq_fairness/reproduce_table6.py   # the fairness table (Table 6), all systems
    python3 harness/analyze_fairness.py --results-dir results     # banking-fairness
    python3 harness/analyze_exp2.py                                # output-safety
    python3 external_panel/decontam_repro/jaccard_decontam_repro.py
    python3 external_panel/decontam_repro/bharatbbq_heldout/check_bharatbbq_heldout.py
    python3 annotation/compute_kappa.py
    python3 external_validation/score_control_validation.py       # expert validation of banking controls

## Fairness (Table 6) release
`results/table6_bharatbbq_fairness/fairness_release/` contains the disclosable policy
configuration, including the 0.70 threshold, frozen per-item output-mode predictions
(catch / over-flag / flip), scoring scripts, expected metrics, and environment/model-version
information. It does not include the deployed discrimination-probe checkpoint. The released
scripts reproduce the reported Table 6 metrics from frozen predictions; end-to-end BankShield
inference remains unavailable because the proprietary LoRA weights and production policy rules are
withheld under the workshop organizers' written commercial/security exception, and the exact
deployed probe checkpoint is unavailable (the base model google/gemma-4-12B-it is public).
Competitor rows are the authors' own runs. `SHA256_MANIFEST.txt` lists a hash for every file.
