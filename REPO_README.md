# BankShield — FinNLP 2026 evaluation artifacts

Reproducibility package for the BankShield FinNLP 2026 evaluation artifacts: a fairness audit of
Indian-language banking safety guardrails (GuardAudit protocol + two banking benchmarks).

Reproduce (no GPU, standard library; openpyxl optional for the form cross-check):

    python3 results/table6_bharatbbq_fairness/reproduce_table6.py     # fairness table (Table 6): 66.7 / 4.9 / 0.0 + action audit
    python3 harness/analyze_fairness.py --results-dir results         # banking (Table 2)
    python3 harness/analyze_exp2.py                                   # output-safety (Table 3)
    python3 external_panel/decontam_repro/jaccard_decontam_repro.py   # decontamination (1,978)
    python3 external_panel/decontam_repro/bharatbbq_heldout/check_bharatbbq_heldout.py
    python3 annotation/compute_kappa.py                               # inter-annotator kappa
    python3 external_validation/score_control_validation.py           # control validation
    python3 external_validation/coded_tier/score_coded_validation.py  # coded-tier expert validation

Every file has a SHA-256 in SHA256_MANIFEST.txt. Proprietary model weights and production
policy rules are withheld under the workshop organizers' written commercial/security exception;
the base model is public. No deployed discrimination-probe checkpoint is included.
