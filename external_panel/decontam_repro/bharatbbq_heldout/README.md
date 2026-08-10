# BharatBBQ held-out verification (the 67% catch)

    python3 check_bharatbbq_heldout.py

Confirms the paper's headline (67% biased-content catch) is a held-out result: the 1,102 BharatBBQ
evaluation items have **0 exact and 0 Jaccard>=0.70** overlap with the 13,622-item union that trains
the output-side discrimination probe (fairness training set + demographic recalibration set).

The training corpus is withheld (deployed financial guard); it ships one-way SHA1-hashed
(`probe_train_hashed_index.json.gz`), so overlap is recomputable without exposing training text.
`bbq_eval_texts.json` is the 1,102 evaluation prompts (also in `results/table6_bharatbbq_fairness/`).
