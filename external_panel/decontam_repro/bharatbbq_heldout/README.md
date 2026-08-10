# BharatBBQ held-out verification (the 67% catch)

    python3 check_bharatbbq_heldout.py

Confirms the paper's headline (67% biased-content catch) is a held-out result: the 1,102 BharatBBQ
evaluation items have **0 exact and 0 Jaccard>=0.70** overlap with the 13,622-item union that trains
the output-side discrimination probe (fairness training set + demographic recalibration set).

The training corpus is withheld (deployed financial guard); it ships one-way SHA1-hashed
(`probe_train_hashed_index.json.gz`), so overlap is recomputable without exposing training text.
`bbq_eval_texts.json` is the 1,102 evaluation prompts (also in `results/table6_bharatbbq_fairness/`).


## ID-level accounting: 1,102 vs the paper's 1,056
The 1,102 figure counts all BharatBBQ evaluation-prompt occurrences pooled across the two per-item result files (biased-catch and benign over-flag); these include 168 texts that appear in both files and 66 records tagged author2. The paper's 1,056 is the unique BharatBBQ pool after deduplication. Both are consistent; the held-out check is run over the pooled occurrences (a superset), so 0/1,102 overlap implies 0 on the 1,056 unique pool.
