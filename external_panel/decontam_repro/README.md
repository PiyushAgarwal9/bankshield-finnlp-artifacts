# Decontamination reproduction (Jaccard-0.70, paper App H) — self-contained, no GPU

    python3 jaccard_decontam_repro.py

Reproduces, per public benchmark, how many items were removed as near-duplicates of BankShield's
training data before the competence panel (Table 4) was scored. Runs in ~2 min on a CPU.

## Why this is shippable without leaking the training corpus
BankShield's training text is withheld (deployed financial guard, RBI/DPDP). Instead of the raw
corpus, this package ships `hashed_train_index_v2.json.gz`:
- `str_hashes` — SHA1 of each unique *normalized training string* (for the exact-duplicate step).
- `toksets` — each training string as a set of SHA1-hashed *tokens* (for the Jaccard-0.70 step).

Hashing is one-way, so the full training text is not directly readable; note that individual token hashes are unsalted truncated SHA-1 and could be recovered by dictionary attack, so this deters casual inspection rather than a determined adversary. Token-set identity is preserved,
so the reviewer recomputes the overlap exactly. Benchmark tokens are hashed the same way at runtime.

## What reproduces
- **Exact-duplicate counts: exact.** Post-exact-dedup set sizes match the paper/`competence_slice_jaccard_result.json`
  to the item (polyguard 3398, harmbench 154, hh-rlhf 11649/12657, xstest 408, jailbreakbench 85,
  toxicchat 10002, or-bench 5000).
- **Jaccard-0.70 near-dup counts: to ~1%.** Total additional near-dups 1978 vs the paper's 1961;
  per set within a handful (harmbench 2, jailbreakbench 2, or-bench 0 exact; polyguard 12 vs 10,
  hh-rlhf 827/938 vs 843/915, xstest 14 vs 12, toxicchat 183 vs 177). The residual differences are
  artifacts of the compliance-safe index rebuild (token-set de-duplication + the common-token index
  cap ordering), not the decontamination logic. The magnitude and direction match App H.

The point the paper makes from these counts — that Hindi-hate and a slice of hh-rlhf overlap the
public supplement and are removed, while BharatBBQ has zero overlap — reproduces fully.
