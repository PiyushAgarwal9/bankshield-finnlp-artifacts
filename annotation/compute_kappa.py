#!/usr/bin/env python3
"""Recompute inter-annotator Cohen's kappa from the two annotators' raw labels.
    python3 compute_kappa.py
Reads kk_annotation_sheet_FILLED.csv (columns: id,set,language,text,human1_label,human2_label).
Reproduces the paper's headline IAA: kappa=0.974, 98.7% agreement (156/158).

Note: one item (fairness_006) carries a non-binary 'ambiguous' label from annotator 1 and is
excluded from the binary safe/unsafe kappa (as in the paper). Including it as a disagreement
gives kappa=0.961 / 98.1% (156/159) -- reported here for full transparency.
"""
import csv, os
from collections import Counter

HERE=os.path.dirname(os.path.abspath(__file__))
rows=list(csv.DictReader(open(os.path.join(HERE,"kk_annotation_sheet_FILLED.csv"))))
def norm(x): return str(x).strip().lower()

def kappa(pairs):
    n=len(pairs); agree=sum(a==b for a,b in pairs)
    labs=set([a for a,_ in pairs]+[b for _,b in pairs])
    c1=Counter(a for a,_ in pairs); c2=Counter(b for _,b in pairs)
    po=agree/n; pe=sum((c1[l]/n)*(c2[l]/n) for l in labs)
    return (po-pe)/(1-pe), agree, n

pairs=[(norm(r["human1_label"]),norm(r["human2_label"])) for r in rows]
binp=[(a,b) for a,b in pairs if a in("safe","unsafe") and b in("safe","unsafe")]
k,ag,n=kappa(binp)
print(f"Inter-annotator Cohen's kappa (binary safe/unsafe): {k:.3f}")
print(f"  agreement: {ag}/{n} = {100*ag/n:.1f}%   [paper: 0.974, 98.7%]")
kk,agg,nn=kappa(pairs)
print(f"Including the 1 non-binary 'ambiguous' item as a disagreement: kappa={kk:.3f}, {agg}/{nn}={100*agg/nn:.1f}%")
