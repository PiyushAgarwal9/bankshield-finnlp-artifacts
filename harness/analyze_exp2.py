"""Exp2 per-category analysis: catch on harmful vs over-block on matched benign,
paired accuracy, per-category + per-language, Wilson CIs."""
import glob, json, math
from collections import defaultdict

def wilson(k,n,z=1.96):
    if n==0: return (0,0,0)
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d
    h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return (p,max(0,c-h),min(1,c+h))
def fmt(k,n):
    p,lo,hi=wilson(k,n); return f"{p*100:5.1f}% [{lo*100:.0f},{hi*100:.0f}] ({k}/{n})"

_RD = "results" if glob.glob("results/*__exp2_output_safety__output.json") else "benchmark/results"
for path in sorted(glob.glob(f"{_RD}/*__exp2_output_safety__output.json")):
    d=json.load(open(path)); recs=d["records"]; fl=lambda r:r["pred"]=="UNSAFE"
    print(f"\n=== {d['system']} (output) ===")
    harm=[r for r in recs if r["metadata"]["kind"]=="harmful"]
    ben =[r for r in recs if r["metadata"]["kind"]=="benign"]
    print(f"  catch harmful:  {fmt(sum(map(fl,harm)),len(harm))}")
    print(f"  over-block benign: {fmt(sum(map(fl,ben)),len(ben))}")
    pairs=defaultdict(dict)
    for r in recs: pairs[r["metadata"]["pair_id"]]["h" if r["metadata"]["kind"]=="harmful" else "b"]=fl(r)
    ok=sum(1 for v in pairs.values() if v.get("h") and not v.get("b"))
    print(f"  paired accuracy: {fmt(ok,len(pairs))}")
    for cat in ["pii_leakage","fraud_enabling","bad_advice","abusive_tone","jailbreak_compliance","selfharm_mishandle"]:
        h=[r for r in harm if r["metadata"]["cat"]==cat]; b=[r for r in ben if r["metadata"]["cat"]==cat]
        print(f"    {cat:22s} catch {sum(map(fl,h))}/{len(h)}  ob {sum(map(fl,b))}/{len(b)}")
    for lang in ["en","hi-en"]:
        h=[r for r in harm if r["metadata"]["lang"]==lang]
        print(f"    lang {lang:6s} catch {sum(map(fl,h))}/{len(h)}")
