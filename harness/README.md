# Analysis scripts

This package is **table-analysis / reproduction only**: it regenerates the paper's table numbers from
the released per-item prediction logs in `../results/`. It does not run the guards themselves (model
weights are not included (see below); the rival adapters in `wrappers/` are included as reference
for how each competitor was invoked).

Run from the package root:

    python3 harness/analyze_fairness.py --results-dir results   # banking-fairness (Table 1)
    python3 harness/analyze_exp2.py                              # output-safety (Table 2)

Other reproductions (see the top-level README):
    python3 external_panel/combine_rivals.py                                   # Table 4
    python3 results/table6_bharatbbq_fairness/reproduce_table6.py              # fairness table
    python3 external_panel/decontam_repro/jaccard_decontam_repro.py            # decontamination
    python3 external_panel/decontam_repro/bharatbbq_heldout/check_bharatbbq_heldout.py
    python3 annotation/compute_kappa.py                                        # inter-annotator kappa

All analyzers use the Python 3.8+ standard library only (no GPU, no third-party packages).

The output-side discrimination probe and its 0.70 threshold ARE included (results/table6_bharatbbq_fairness/fairness_release/); the LoRA adapter weights and production policy rules are not, so the released per-item logs reproduce the reported metrics but the package does not regenerate BankShield predictions end-to-end.
