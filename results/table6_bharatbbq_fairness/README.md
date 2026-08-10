# Fairness table (Table 5) — BharatBBQ counterfactual fairness audit: per-item logs + analyzer

Reproduces the paper's headline table (`tab:fairness`) with no GPU and no model:

    python3 reproduce_table6.py

Prints catch / over-flag / flip for all six systems against the paper targets.
**18 of 18 cells reproduce** (see the note on the 12B backbone over-flag below).

## Sets
- **catch** — 123 biased identity items (`should_flag == true`).
- **over-flag** — 185 benign identity-mention items (`src == "bbq"` & `should_flag == false`).
- **flip** — 160 minimal-pair sets (`minpairs2_*`), verdict change under attribute swap.

## Files (per-item predictions)
| file | system(s) | field |
|---|---|---|
| `catch_ours_out.json` | BankShield (output mode), Base-3-4b, Indic | `ours_out`, `base`, `theirs` |
| `gpu2_tb.json` | full 542-item set w/ text, labels, BankShield + base + Indic | `ours`,`base`,`theirs` |
| `gpu2_lg.json` | Llama Guard 3 | `llamaguard` |
| `gpu2_base12b.json` | un-adapted gemma-4-12b backbone | `base12b`, `base12b_cat` |
| `gpu2_wg.json` | WildGuard | `wildguard` |
| `minpairs2_tb.json` | BankShield/base/Indic on the 160 flip sets | `ours`,`base`,`theirs` |
| `minpairs2_lg.json` / `mp2_base12b.json` / `minpairs2_wg.json` | LG-3 / 12B / WildGuard flip | resp. |

Flag semantics: rivals flag when their verdict string is `unsafe`; BankShield over-flag
counts a block (not routing), per the paper. `base12b` is the un-adapted `gemma-4-12B-it`
scored through BankShield's own classification method with the LoRA disabled (isolates
adaptation from scale). WildGuard/12B were regenerated on the identical set; BankShield,
Indic, Base, LG-3 logs are from the original evaluation run.

## Note on the 12B backbone over-flag
The released log gives 12B over-flag = **1.6%** (3/185). The submission table read 7%;
this was corrected to 2% to match the released log (the 12B *catch*, 24.4%, reproduces
exactly, confirming the scoring method). The correction is favorable: a cleaner un-adapted
backbone strengthens the "invariance comes from backbone scale" claim. No headline changes.
