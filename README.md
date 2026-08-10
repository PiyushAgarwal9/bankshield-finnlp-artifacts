# BankShield — reproducibility package

Per-item prediction logs and analyzers behind the paper's tables.

## Layout
- `paper/` — submission PDF and LaTeX source.
- `data/` — frozen benchmark sets and prompt sheets.
- `results/` — per-item prediction logs, `<system>__<benchmark>__<direction>.json`.
- `harness/` — analysis scripts (regenerate tables from released logs; adapters under `harness/wrappers/` are reference code).
- `external_panel/` — external-benchmark competence panel + decontamination reproduction.
- `annotation/` — inter-annotator-agreement package.

## Reproduce tables (no GPU, no model)
    python3 results/table6_bharatbbq_fairness/reproduce_table6.py   # the fairness table (Table 5), all systems
    python3 harness/analyze_fairness.py --results-dir results     # banking-fairness
    python3 harness/analyze_exp2.py                                # output-safety
    python3 external_panel/decontam_repro/jaccard_decontam_repro.py
    python3 external_panel/decontam_repro/bharatbbq_heldout/check_bharatbbq_heldout.py
    python3 annotation/compute_kappa.py
    python3 external_validation/score_control_validation.py       # expert validation of banking controls

## Fairness (the fairness table (Table 5)) release
`results/table6_bharatbbq_fairness/fairness_release/` contains the output-side discrimination
probe with its 0.70 decision threshold, the scoring script, the per-item output-mode predictions
(catch / over-flag / flip), expected metric values, and environment/model-version info. These make
the guard-scoring stage reproducible from the frozen predictions; `reproduce_table6.py` recomputes
every reported Table 5 value with no GPU.

The proprietary LoRA adapter weights and production policy rules are not included, under the
workshop commercial/security exception (request pending with organizers), so the package does not regenerate BankShield predictions
end-to-end (the base model google/gemma-4-12B-it is public). Competitor rows are the authors' own
runs. `SHA256_MANIFEST.txt` lists a hash for every file.
