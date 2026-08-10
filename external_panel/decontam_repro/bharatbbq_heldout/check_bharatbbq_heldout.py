#!/usr/bin/env python3
"""Verify the headline 67% catch is HELD-OUT: the BharatBBQ evaluation items are disjoint from
the data that trains BankShield's output-side discrimination probe.
    python3 check_bharatbbq_heldout.py     (no GPU, no model, no training corpus)

The probe trains on the UNION of two in-house synthetic sets (fairness training + demographic
recalibration), 13,622 safe/discrimination items. That corpus is withheld (deployed financial guard);
here it ships one-way-hashed (SHA1) as probe_train_hashed_index.json.gz -- str_hashes for exact-dup,
tok-sets for Jaccard>=0.70 -- so overlap recomputes exactly without exposing training text.
Expected result: 0 exact and 0 near-duplicate overlaps across all 1,102 evaluation items.
"""
import json, re, unicodedata, math, hashlib, gzip, os
from collections import Counter, defaultdict
HERE=os.path.dirname(os.path.abspath(__file__))
PUNC=re.compile(r"[^\w\s]", re.UNICODE)
def norm(t):
    t=unicodedata.normalize("NFC",str(t)).lower(); t=PUNC.sub(" ",t)
    return re.sub(r"\s+"," ",t).strip()
def h(tok): return hashlib.sha1(tok.encode("utf-8")).hexdigest()[:10]
def hs(s): return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]

idx=json.load(gzip.open(os.path.join(HERE,"probe_train_hashed_index.json.gz")))
train_str=set(idx["str_hashes"]); train_toksets=[set(x) for x in idx["toksets"]]
inv=defaultdict(list)
for i,ts in enumerate(train_toksets):
    for tk in ts:
        if len(inv[tk])<=25000: inv[tk].append(i)
print(f"probe-train union: {idx['n_train_items']} items ({', '.join(idx['sources'])}); hashed token-sets {len(train_toksets)}")

ev=[t for t in json.load(open(os.path.join(HERE,"bbq_eval_texts.json"))) if t]
print(f"BharatBBQ evaluation items: {len(ev)}")
def jhit(A):
    if not A: return 0
    need=math.ceil(0.70*len(A)); cand=Counter()
    for tk in A:
        for i in inv.get(tk,()): cand[i]+=1
    for i,ov in cand.items():
        if ov>=need and ov/len(A|train_toksets[i])>=0.70: return 1
    return 0
exact=sum(1 for t in ev if hs(norm(t)) in train_str)
near=sum(jhit(set(h(x) for x in norm(t).split() if x)) for t in ev)
print(f"\nexact-normalized overlaps: {exact}/{len(ev)}")
print(f"Jaccard>=0.70 near-dups:    {near}/{len(ev)}")
print("\nHELD-OUT CONFIRMED." if exact==0 and near==0 else "\nWARNING: overlap found.")
