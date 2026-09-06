# Analysis scripts

This package is **table-analysis / reproduction only**: it regenerates the paper's table numbers from
the released per-item prediction logs in `../results/`. It does not run the guards themselves (model
weights are not included (see below); the rival adapters in `wrappers/` are included as reference
for how each competitor was invoked).

Run from the package root:

    python3 harness/analyze_fairness.py --results-dir results   # banking-fairness (Table 2)
    python3 harness/analyze_exp2.py                              # output-safety (Table 3)

Other reproductions (see the top-level README):
    python3 external_panel/combine_rivals.py                                   # Table 5 (external competence panel)
    python3 results/table6_bharatbbq_fairness/reproduce_table6.py              # fairness table
    python3 external_panel/decontam_repro/jaccard_decontam_repro.py            # decontamination
    python3 external_panel/decontam_repro/bharatbbq_heldout/check_bharatbbq_heldout.py
    python3 annotation/compute_kappa.py                                        # inter-annotator kappa

All analyzers use the Python 3.8+ standard library only (no GPU, no third-party packages).

The fairness release ships the disclosable policy configuration (including the 0.70 threshold), frozen per-item output-mode predictions, scoring scripts, expected metrics and environment information; it does NOT include the deployed discrimination-probe checkpoint. The released per-item logs reproduce the reported metrics from frozen predictions; end-to-end BankShield inference remains unavailable because the LoRA adapter weights and production policy rules are withheld under the organizers' written commercial/security exception, and the exact deployed probe checkpoint is unavailable. Not all inference code or model components are released.
