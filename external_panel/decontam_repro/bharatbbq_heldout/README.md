# BharatBBQ held-out verification (the 67% catch)

    python3 check_bharatbbq_heldout.py

Confirms the paper's headline (67% biased-content catch) is a held-out result: the 1,102 pooled
BharatBBQ prompt occurrences (934 unique; 868 target prompts after excluding 66 `author2` records)
have **0 exact and 0 Jaccard>=0.70** overlap with the 13,622-item union that trains the output-side
discrimination probe (fairness training set + demographic recalibration set).

The training corpus is withheld (deployed financial guard); it ships one-way SHA1-hashed
(`probe_train_hashed_index.json.gz`), so overlap is recomputable without exposing training text.
`bbq_eval_texts.json` is the 1,102 pooled prompt occurrences (also in `results/table6_bharatbbq_fairness/`).


## ID-level accounting: 1,102 pooled occurrences vs the paper's 1,056 sample

These two numbers count different things and are not a deduplication of each other.

**1,056** is the *upstream* BharatBBQ sample the fairness audit draws from (paper Sec. "Fairness
data" and the decontamination appendix): from this pool we draw 185 benign identity-mention items
and 123 biased items and construct 160 minimal-pair counterfactual sets. It is a sampling pool, not
an evaluated-item count.

**1,102** is the number of BharatBBQ evaluation-prompt *occurrences* actually scored, pooled across
the two per-item result files (biased-catch and benign over-flag) that this held-out check reads.
Of these 1,102 occurrences:
- 168 are duplicate occurrences (texts appearing in both files), leaving **934 unique texts**
  (1,102 - 168 = 934);
- 66 are records tagged `author2` (a second annotator pass); excluding them leaves **868 unique
  texts** (934 - 66 = 868).

The held-out decontamination check runs over all 1,102 pooled occurrences (a superset of the unique
evaluated set), so **0/1,102 exact and Jaccard>=0.70 overlap implies 0 overlap on every evaluated
item**. The IDs of every scored item are in the per-item result files under
`results/table6_bharatbbq_fairness/` and in `bbq_eval_texts.json` here; the 1,056 figure refers to
the upstream sample from which those items were drawn, not to a deduplicated evaluation count.
