import json, re, unicodedata, math, glob, os, time
from collections import Counter, defaultdict
PUNC=re.compile(r"[^\w\s]", re.UNICODE)
def norm(t):
    t=unicodedata.normalize("NFC",str(t)).lower(); t=PUNC.sub(" ",t)
    return re.sub(r"\s+"," ",t).strip()
def toks(t): return set(norm(t).split())
KEYS=("text","prompt","input","utterance","content","question","message","response","output")
STAGE="/tmp/l40stage"
# --- training: exact-normalized set + tokenized sets + inverted index ---
train_norm=set(); train_toksets=[]; inv=defaultdict(list)
t0=time.time()
for f in ["gemma_sft_train","unified_train","v3_train","fairness_trainset_final_10k","abuse_probe_train_v2","abuse_probe_train","recalib_multi"]:
    p=os.path.join(STAGE,f+".jsonl")
    if not os.path.exists(p): continue
    for line in open(p):
        line=line.strip()
        if not line: continue
        try: o=json.loads(line)
        except: continue
        for k in KEYS:
            if isinstance(o.get(k),str) and o[k].strip():
                n=norm(o[k]); train_norm.add(n)
                break
# build token sets + inverted index from UNIQUE normalized train strings
for n in train_norm:
    ts=set(n.split())
    if not ts: continue
    idx=len(train_toksets); train_toksets.append(ts)
    for tok in ts:
        if len(inv[tok])<=25000:  # cap ultra-common tokens (rare tokens still catch near-dups)
            inv[tok].append(idx)
print("train: %d unique-norm, index built in %.1fs" % (len(train_norm), time.time()-t0), flush=True)

UNSAFE=("unsafe","harmful","1","true")
DATASETS=["polyguard","harmbench","hhrlhf_harmful","hhrlhf_safe","xstest","jailbreakbench","toxicchat","or_bench"]
def jaccard_hit(A):
    if not A: return False
    need=math.ceil(0.70*len(A))
    cand=Counter()
    for tok in A:
        for idx in inv.get(tok,()): cand[idx]+=1
    for idx,ov in cand.items():
        if ov<need: continue
        B=train_toksets[idx]
        j=ov/len(A|B)
        if j>=0.70: return True
    return False

print("\n%-16s %8s %8s %8s %8s" % ("dataset","raw","exact-uniq","jacc-dup","jacc%"), flush=True)
tot_exact=tot_jacc=0
for ds in DATASETS:
    d=json.load(open(os.path.join(STAGE,"bench_results",ds+".json")))
    recs=d if isinstance(d,list) else (d.get("results") or list(d.values()))
    raw=[(r["text"], str(r.get("gold")).lower() in UNSAFE) for r in recs if r.get("text")]
    # exact-normalize dedup (what the panel already did)
    exact=[(t,g) for t,g in raw if norm(t) not in train_norm]
    # additional Jaccard>=0.70 near-dups among exact survivors
    jd=0; jd_u=jd_s=0
    for t,g in exact:
        if jaccard_hit(toks(t)):
            jd+=1
            if g: jd_u+=1
            else: jd_s+=1
    tot_exact+=len(exact); tot_jacc+=jd
    print("%-16s %8d %8d %8d %7.2f%%  (harm %d / safe %d)" % (ds,len(raw),len(exact),jd,100*jd/max(len(exact),1),jd_u,jd_s), flush=True)
print("\nTOTAL exact-uniq=%d  additional-jaccard-dups=%d  (%.2f%% of scored pool)" % (tot_exact,tot_jacc,100*tot_jacc/max(tot_exact,1)), flush=True)
