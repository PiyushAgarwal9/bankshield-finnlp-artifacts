# Output-safety benchmark: v1 (evaluated) vs v2 (corrected)

- `output_safety_66pairs_v1_original.jsonl` — the benchmark AS EVALUATED. Table 3 and every competitor
  log in `results/*__exp2_output_safety*.json` were run on this version. Its 11 benign distress controls
  cite a crisis helpline (KIRAN) that has since been superseded.
- `output_safety_66pairs_v2_corrected.jsonl` — the corrected release: the 11 controls now cite the current
  national line (Tele-MANAS, 14416). Only those 11 prompts differ from v1.

Re-running BankShield on v2 (`results/bankshield__exp2_output_safety_v2.json`) marginally lowers its
over-block (65.2% vs 68.2%; catch unchanged at 95.5%). The competitor systems could not be re-run, so
Table 3 in the paper reports the consistent v1 all-systems evaluation; the v2 rerun and the correction
are disclosed in the paper's Limitations. Use v2 for any downstream work.

## Banking benchmark: 256 (provenance) vs 250 (evaluated)
`banking_fairness_256.jsonl` is retained for provenance. Six explicit-tier prompts had malformed
templates ("a Muslim customers", "woman community"); `banking_fairness_250_evaluated.jsonl` excludes
them and is the set the analyzers score (explicit catch over 26 = 32 minus 6). Table 2's explicit
column already uses the 250-item set.
