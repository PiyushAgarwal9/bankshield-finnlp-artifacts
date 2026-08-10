# Coded-tier validation (all 40 coded pairs) — COMPLETED

Two independent human domain experts blind-labeled all 40 coded pairs (80 items: 40 discriminatory
+ 40 matched legitimate controls), covering the exact tier behind the 75.0% control false-flag rate.

Run: `python3 score_coded_validation.py`

Result: both experts accepted essentially every legitimate control (40/40 and 39/40), so the controls
are expert-validated and the 75.0% is a genuine over-flag rate. Agreement with our labels k=0.80 and
0.75 (expert-expert k=0.90); they confirmed 32/40 and 31/40 of the coded-discriminatory items (the
coded-proxy tier is subtle, so a few read as borderline).

- `coded_validation_results.json` per-item ratings (experts anonymized A/B) + our label, tier (kind),
  pair_id, text SHA-256.
- `coded_validation_sheet.xlsx`   the blank blind instrument (instructions on the second tab).
- `coded_gold.json`               our labels + tier + text SHA-256.
- `score_coded_validation.py`     reproduces the numbers above from the released ratings.

Experts are volunteer bilingual professionals who worked independently and blind, with informed
consent and no compensation; identities anonymized.
