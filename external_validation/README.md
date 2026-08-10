# Control validation (banking legitimate/discriminatory labels)

Blind validation of a 120-item stratified sample. Two annotators are released here:
- `reviewer_clarified` — used the clarified construct (an objective documented reason makes an
  incidental protected-attribute mention legitimate); agreed with our labels at Cohen's k=0.93.
- `initial_reviewer`   — used earlier, stricter instructions (any protected-attribute mention =
  discriminatory); rated a 110-item subset; k=0.03.

The disagreement is construct-driven, so control validity is construct-sensitive. This 120-item
sample covered only 8 of the 36 coded-tier controls behind the 75.0% control false-flag rate;
`coded_tier/` is a dedicated instrument covering all 40 coded pairs (see coded_tier/README.md).

Run: `python3 score_control_validation.py`  (also self-verifies the itNNN -> benchmark-row mapping).

`control_validation.json` holds per-item ratings; `PROTOCOL.md` gives the exact instructions;
each record carries `benchmark_row_index` and `text_sha256` mapping it to data/banking_fairness_256.jsonl.
Reviewers are volunteer professionals who worked independently and blind, with informed consent and no
compensation; identities anonymized.
