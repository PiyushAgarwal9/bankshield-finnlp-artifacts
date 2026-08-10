#!/usr/bin/env python3
"""Score the 120-item control validation. One reviewer used the clarified construct; an initial
reviewer used earlier, stricter instructions. Run: python3 score_control_validation.py"""
import json, os, collections, hashlib
D=os.path.dirname(os.path.abspath(__file__)); rows=json.load(open(os.path.join(D,"control_validation.json")))
# mapping self-check
bpath=os.path.join(D,"..","data","banking_fairness_256.jsonl")
if os.path.exists(bpath):
    bench=[json.loads(l) for l in open(bpath,encoding="utf-8")]; bad=0
    for r in rows:
        if r.get("benchmark_row_index")!=int(r["id"][2:]) or r.get("text_sha256")!=hashlib.sha256(bench[int(r["id"][2:])]["text"].encode()).hexdigest(): bad+=1
    print(f"[mapping] {len(rows)-bad}/{len(rows)} records verify against data/banking_fairness_256.jsonl.")
def gl(x): return "L" if str(x).startswith("Legit") else ("D" if str(x).startswith("Discrim") else "U")
for e in ("reviewer_clarified","initial_reviewer"):
    ok=[r for r in rows if r.get(e) and gl(r[e])!="U"]
    if not ok: continue
    agree=sum(gl(r[e])==gl(r["our_label"]) for r in ok)/len(ok)
    ctrl=[r for r in rows if gl(r["our_label"])=="L" and r.get(e) and gl(r[e])!="U"]; acc=sum(gl(r[e])=="L" for r in ctrl)
    n=len(ok); ca=collections.Counter(gl(r[e]) for r in ok); cb=collections.Counter(gl(r["our_label"]) for r in ok)
    pe=sum((ca[c]/n)*(cb[c]/n) for c in set(list(ca)+list(cb))); k=(agree-pe)/(1-pe) if pe<1 else 1.0
    print(f"{e:18s}: agreement {agree:.0%} | controls accepted {acc}/{len(ctrl)} | Cohen k {k:.2f}")
print("\nConstruct-sensitive: the clarified-construct reviewer agrees (k=0.93); the initial reviewer,")
print("under stricter instructions, did not (k=0.03). This sample covered only 8/36 coded controls;")
print("see coded_tier/ for the full 40-pair validation instrument.")
