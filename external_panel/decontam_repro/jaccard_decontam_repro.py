#!/usr/bin/env python3
"""Self-contained reproduction of the Jaccard-0.70 decontamination (paper App H).
    python3 jaccard_decontam_repro.py
Reproduces, per public benchmark, the count of items removed as near-duplicates of BankShield's
training data. No GPU, no model, and NO access to the raw training corpus required.

How the training side is shipped compliance-safely: BankShield's training text is withheld (deployed
financial guard). Instead we ship hashed_train_index.json -- each unique normalized training string as
a set of SHA1-hashed tokens. Hashing is one-way, so the full text is not directly readable (individual token hashes are unsalted truncated SHA-1 and could be dictionary-recovered; this deters casual inspection, not a determined adversary), but preserves
token-set identity, so the Jaccard overlap is recomputed EXACTLY. This script hashes the public benchmark
tokens the same way and compares. Result matches competence_slice_jaccard_result.json / App H.
"""
import json, re, unicodedata, math, hashlib, os, gzip
from collections import Counter, defaultdict

HERE=os.path.dirname(os.path.abspath(__file__))
PUNC=re.compile(r"[^\w\s]", re.UNICODE)
def norm(t):
    t=unicodedata.normalize("NFC",str(t)).lower(); t=PUNC.sub(" ",t)
    return re.sub(r"\s+"," ",t).strip()
def h(tok): return hashlib.sha1(tok.encode("utf-8")).hexdigest()[:10]

def hs(s): return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]
print("loading hashed training index ...", flush=True)
idx=json.load(gzip.open(os.path.join(HERE,"hashed_train_index_v2.json.gz")))
train_str=set(idx["str_hashes"])          # exact full-normalized-string dedup (matches original)
train_toksets=[set(x) for x in idx["toksets"]]
inv=defaultdict(list)
for i,ts in enumerate(train_toksets):
    for tok in ts:
        if len(inv[tok])<=25000: inv[tok].append(i)
print("hashed training token-sets: %d" % len(train_toksets), flush=True)

def jaccard_hit(A):
    if not A: return False
    need=math.ceil(0.70*len(A)); cand=Counter()
    for tok in A:
        for i in inv.get(tok,()): cand[i]+=1
    for i,ov in cand.items():
        if ov<need: continue
        B=train_toksets[i]
        if ov/len(A|B)>=0.70: return True
    return False

UNSAFE=("unsafe","harmful","1","true")
DATASETS=["polyguard","harmbench","hhrlhf_harmful","hhrlhf_safe","xstest","jailbreakbench","toxicchat","or_bench"]
print("\n%-16s %8s %10s %9s %7s" % ("dataset","raw","exact-uniq","jacc-dup","jacc%"), flush=True)
tot_exact=tot_jacc=0
for ds in DATASETS:
    d=json.load(open(os.path.join(HERE,"bench_results",ds+".json")))
    recs=d if isinstance(d,list) else (d.get("results") or list(d.values()))
    raw=[(r["text"], str(r.get("gold")).lower() in UNSAFE) for r in recs if r.get("text")]
    exact=[(t,g) for t,g in raw if hs(norm(t)) not in train_str]
    jd=sum(1 for t,g in exact if jaccard_hit(set(h(x) for x in norm(t).split() if x)))
    tot_exact+=len(exact); tot_jacc+=jd
    print("%-16s %8d %10d %9d %6.2f%%" % (ds,len(raw),len(exact),jd,100*jd/max(len(exact),1)), flush=True)
print("\nTOTAL exact-uniq=%d  additional-jaccard-dups=%d (%.2f%%)" % (tot_exact,tot_jacc,100*tot_jacc/max(tot_exact,1)), flush=True)
