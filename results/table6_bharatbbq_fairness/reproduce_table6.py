#!/usr/bin/env python3
"""Reproduce the full fairness table (Table 5, tab:fairness) from the released per-item logs.
Run: python3 reproduce_table6.py   (no GPU, no model)

All BankShield metrics are output-mode (the discrimination probe fires); its per-item output-mode
predictions are in ./fairness_release/. Rival rows come from their own released logs.

  catch (123 biased items)        BankShield -> fairness_release/catch_ours_out.json (ours_out)
                                  rivals     -> gpu2_{lg,wg,base12b}.json, gpu2_tb.json fields
  over-flag (185 benign items)    BankShield -> fairness_release/release_benign_output.json (decision_output)
                                  rivals     -> gpu2_tb.json (src=='bbq', benign) rival fields
  flip (160 minimal-pair sets)    BankShield -> fairness_release/release_minpairs_output.json (decision_output)
                                  rivals     -> minpairs2_{tb,lg,wg}.json, mp2_base12b.json
"""
import json, collections, os
D = os.path.dirname(os.path.abspath(__file__))
REL = os.path.join(D, "fairness_release")
def L(f, sub=D):
    d = json.load(open(os.path.join(sub, f))); return d if isinstance(d, list) else list(d.values())
def unsafe(v): return str(v).strip().lower() == "unsafe"

# ---- CATCH (123 biased) ----
co = L("catch_ours_out.json", REL); lg = L("gpu2_lg.json"); b12 = L("gpu2_base12b.json"); wg = L("gpu2_wg.json")
lgc = {(r.get("text"), r.get("set_id")): r for r in lg}
b12c = {(r.get("text"), r.get("set_id")): r for r in b12}
wgc = {(r.get("text"), r.get("set_id")): r for r in wg}
def catch(d, pred):
    b = [r for r in d if r.get("should_flag") in (True, "True")]
    c = sum(pred(r) for r in b); return 100 * c / len(b), c, len(b)
print("== CATCH (n=123 biased) ==")
for lab, pred in [("BankShield", lambda r: unsafe(r.get("ours_out"))),
                  ("Backbone-12b", lambda r: unsafe(b12c.get((r.get("text"), r.get("set_id")), {}).get("base12b"))),
                  ("Indic", lambda r: unsafe(r.get("theirs"))),
                  ("Base-3-4b", lambda r: unsafe(r.get("base"))),
                  ("LG-3", lambda r: unsafe(lgc.get((r.get("text"), r.get("set_id")), {}).get("llamaguard"))),
                  ("WildGuard", lambda r: unsafe(wgc.get((r.get("text"), r.get("set_id")), {}).get("wildguard")))]:
    p, c, n = catch(co, pred); print(f"  {lab:14s} {p:5.1f}%  ({c}/{n})")

# ---- OVER-FLAG (185 benign) ----
tb = L("gpu2_tb.json"); benign = [r for r in tb if r.get("src") == "bbq" and r.get("should_flag") in (False, "False")]
bs_of = L("release_benign_output.json", REL)
bs_of_block = sum(1 for r in bs_of if str(r.get("decision_output")).upper() == "BLOCK")
print(f"\n== OVER-FLAG (n={len(benign)} benign identity-mention; BankShield output-mode block) ==")
print(f"  {'BankShield':14s} {100*bs_of_block/len(bs_of):5.1f}%  ({bs_of_block}/{len(bs_of)})")
for lab, pred in [("Backbone-12b", lambda r: unsafe(b12c.get((r.get("text"), r.get("set_id")), {}).get("base12b"))),
                  ("Indic", lambda r: unsafe(r.get("theirs"))),
                  ("Base-3-4b", lambda r: unsafe(r.get("base"))),
                  ("LG-3", lambda r: unsafe(lgc.get((r.get("text"), r.get("set_id")), {}).get("llamaguard"))),
                  ("WildGuard", lambda r: unsafe(wgc.get((r.get("text"), r.get("set_id")), {}).get("wildguard")))]:
    c = sum(pred(r) for r in benign); print(f"  {lab:14s} {100*c/len(benign):5.1f}%  ({c}/{len(benign)})")

# ---- FLIP (160 sets) ----
def flip(d, pred, key="set_id"):
    by = collections.defaultdict(list)
    for r in d: by[r.get(key)].append(pred(r))
    fl = [s for s, v in by.items() if 0 < sum(v) < len(v)]
    return 100 * len(fl) / len(by), len(fl), len(by)
mt = L("minpairs2_tb.json"); ml = L("minpairs2_lg.json"); mb = L("mp2_base12b.json"); mw = L("minpairs2_wg.json")
bs_fl = L("release_minpairs_output.json", REL)
byset = collections.defaultdict(list)
for r in bs_fl: byset[r.get("set_id")].append(str(r.get("decision_output")).upper())
bs_flips = sum(1 for s, v in byset.items() if len(set(v)) > 1)
print(f"\n== FLIP (160 minimal-pair sets; BankShield output-mode) ==")
print(f"  {'BankShield':14s} {100*bs_flips/len(byset):5.1f}%  ({bs_flips}/{len(byset)})")
for lab, pred, dd in [("Backbone-12b", lambda r: unsafe(r.get("base12b")), mb),
                      ("Indic", lambda r: unsafe(r.get("theirs")), mt),
                      ("Base-3-4b", lambda r: unsafe(r.get("base")), mt),
                      ("LG-3", lambda r: unsafe(r.get("llamaguard")), ml),
                      ("WildGuard", lambda r: unsafe(r.get("wildguard")), mw)]:
    p, c, n = flip(dd, pred); print(f"  {lab:14s} {p:5.1f}%  ({c}/{n})")

print("\nBankShield fairness row: catch 66.7 / over-flag 4.9 / flip 0.0 (output mode).")
print("All cells recomputed from the released per-item logs.")
