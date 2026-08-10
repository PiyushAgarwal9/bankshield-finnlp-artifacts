import os as _os
_D=_os.environ.get("DATA_DIR",".")
import json, re, unicodedata, urllib.request, random, math, time
from collections import Counter, defaultdict
random.seed(11)
PUNC = re.compile(r"[^\w\s]", re.UNICODE)
def norm(t):
    t = unicodedata.normalize("NFC", str(t)).lower(); t = PUNC.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()
KEYS = ("text","prompt","input","utterance","content","question","message","response","output")
tn = set()
for f in ["gemma_sft_train","unified_train","v3_train","fairness_trainset_final_10k","abuse_probe_train_v2","abuse_probe_train","recalib_multi"]:
    try:
        for line in open(_D+"/%s.jsonl" % f):
            line=line.strip()
            if not line: continue
            try: o=json.loads(line)
            except: continue
            for k in KEYS:
                if isinstance(o.get(k),str) and o[k].strip(): tn.add(norm(o[k])); break
    except FileNotFoundError: pass
# token index for Jaccard>=0.70 (GuardAudit 5.1 standard)
toksets=[]; inv=defaultdict(list)
for n in tn:
    ts=set(n.split())
    if not ts: continue
    i=len(toksets); toksets.append(ts)
    for tk in ts:
        if len(inv[tk])<=25000: inv[tk].append(i)
def jacc_hit(A):
    if not A: return False
    need=math.ceil(0.70*len(A)); cand=Counter()
    for tk in A:
        for i in inv.get(tk,()): cand[i]+=1
    for i,ov in cand.items():
        if ov<need: continue
        if ov/len(A|toksets[i])>=0.70: return True
    return False
print("train-norm set: %d" % len(tn), flush=True)
def wilson(k,n,z=1.96):
    if n==0: return (0,0)
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return (round(100*max(0,c-h),1), round(100*min(1,c+h),1))
def scan(t):
    req=urllib.request.Request("http://localhost:8000/v1/scan",
        data=json.dumps({"text":t,"direction":"input"}).encode(), headers={"Content-Type":"application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=30).read())
UNSAFE=("unsafe","harmful","1","true"); EXT=["polyguard","harmbench","hhrlhf_harmful","hhrlhf_safe","xstest","jailbreakbench","toxicchat","or_bench"]; N=250
out={}; t0=time.time()
print("%-16s %5s | IG_rec  IG_ob | AWS_rec AWS_ob | jacc_removed" % ("dataset","n"), flush=True)
for ds in EXT:
    try: d=json.load(open(_D+"/bench_results/%s.json"%ds))
    except FileNotFoundError: continue
    recs=d if isinstance(d,list) else (d.get("results") or list(d.values()))
    exact=[r for r in recs if r.get("text") and norm(r["text"]) not in tn]
    removed=0; recs2=[]
    for r in exact:
        if jacc_hit(set(norm(r["text"]).split())): removed+=1
        else: recs2.append(r)
    unsafe=[r for r in recs2 if str(r.get("gold")).lower() in UNSAFE]
    safe=[r for r in recs2 if str(r.get("gold")).lower() not in UNSAFE]
    random.shuffle(unsafe); random.shuffle(safe)
    samp=unsafe[:N]+safe[:N]
    ig_u=ig_s=aws_u=aws_s=nu=ns=0
    for r in samp:
        g=str(r.get("gold")).lower() in UNSAFE
        try: igf=scan(r["text"]).get("decision","ALLOW")!="ALLOW"
        except Exception: continue
        awsf=str(r.get("aws")).lower() in ("true","1","unsafe","block")
        if g: nu+=1; ig_u+=igf; aws_u+=awsf
        else: ns+=1; ig_s+=igf; aws_s+=awsf
    rlo,rhi=wilson(ig_u,nu); olo,ohi=wilson(ig_s,ns)
    print("%-16s %5d | %5.1f[%s-%s] %5.1f[%s-%s] | %5.1f %5.1f | %d" % (ds,nu+ns,100*ig_u/max(nu,1),rlo,rhi,100*ig_s/max(ns,1),olo,ohi,100*aws_u/max(nu,1),100*aws_s/max(ns,1),removed), flush=True)
    out[ds]={"n":nu+ns,"ig_recall":[ig_u,nu],"ig_overblock":[ig_s,ns],"aws_recall":[aws_u,nu],"aws_overblock":[aws_s,ns],"jaccard_removed":removed}
json.dump(out, open(_D+"/competence_slice_jaccard_result.json","w"), indent=1)
print("DONE (%ds)"%(time.time()-t0), flush=True)
