#!/usr/bin/env python3
"""Combine BankShield + AWS competence slice with the L40 rival panel into one
paired recall / over-block table (pooled across the 6 decontaminated external
sets, Wilson 95% CIs). Emits markdown + a LaTeX booktabs table for paper 6."""
import json, math, os, glob

def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0, 0.0)
    p = k/n; d = 1+z*z/n
    c = (p+z*z/(2*n))/d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return (100*p, round(100*max(0, c-h), 1), round(100*min(1, c+h), 1))

DIR = os.path.dirname(os.path.abspath(__file__))

# --- BankShield + AWS from the frozen competence slice ---
comp = json.load(open(os.path.join(DIR, "competence_slice_jaccard_result.json")))
def pool_from_comp(recall_key, ob_key):
    rk = rn = ok = on = 0
    for ds, v in comp.items():
        a, b = v[recall_key]; c, e = v[ob_key]
        rk += a; rn += b; ok += c; on += e
    return (rk, rn, ok, on)

systems = {}  # name -> (rk, rn, ok, on)
systems["BankShield (ours)"] = pool_from_comp("ig_recall", "ig_overblock")
systems["AWS Bedrock Guardrails"] = pool_from_comp("aws_recall", "aws_overblock")

# --- rivals from rivals_<name>.json (pulled from L40) ---
LABELS = {
    "llama_guard_3": "Llama-Guard-3-8B",
    "llama_guard_4": "Llama-Guard-4-12B",
    "wildguard": "WildGuard-7B",
    "polyguard": "PolyGuard-Ministral",
    "qwen3guard": "Qwen3Guard-Gen-8B",
    "l3cube": "L3Cube-IndicGuard (Gemma-3-4B LoRA)",
}
for path in sorted(glob.glob(os.path.join(DIR, "rivals_*.json"))):
    name = os.path.basename(path)[len("rivals_"):-len(".json")]
    d = json.load(open(path))
    sane = d.get("_sanity_ok", None)
    rk = rn = ok = on = 0
    for ds, v in d.items():
        if ds.startswith("_"): continue
        a, b = v["recall"]; c, e = v["overblock"]
        rk += a; rn += b; ok += c; on += e
    lbl = LABELS.get(name, name) + ("" if sane else "  [SANITY-FAIL: verify template]")
    systems[lbl] = (rk, rn, ok, on)

# --- print pooled comparison ---
print("\n=== POOLED recall / over-block across decontaminated external sets ===")
print("%-42s %-24s %-24s" % ("System", "Recall (harm caught)", "Over-block (safe flagged)"))
print("-"*92)
rows = []
for name, (rk, rn, ok, on) in systems.items():
    rp, rlo, rhi = wilson(rk, rn)
    op, olo, ohi = wilson(ok, on)
    rows.append((name, rp, rlo, rhi, rk, rn, op, olo, ohi, ok, on))
    print("%-42s %5.1f [%4.1f-%4.1f] (%5d/%-5d) %5.1f [%4.1f-%4.1f] (%4d/%-5d)"
          % (name, rp, rlo, rhi, rk, rn, op, olo, ohi, ok, on))

# --- LaTeX ---
def esc(s): return s.replace("&", "\\&").replace("_", "\\_")
tex = ["\\begin{tabular}{lcc}", "\\toprule",
       "System & Recall (\\%) & Over-block (\\%) \\\\", "\\midrule"]
for name, rp, rlo, rhi, rk, rn, op, olo, ohi, ok, on in rows:
    rcell = "--" if rn == 0 else f"{rp:.1f} {{\\footnotesize[{rlo:.0f}--{rhi:.0f}]}}"
    ocell = "--" if on == 0 else f"{op:.1f} {{\\footnotesize[{olo:.0f}--{ohi:.0f}]}}"
    nm = esc(name)
    if name.startswith("BankShield"): nm = "\\textbf{" + nm + "}"
    tex.append(f"{nm} & {rcell} & {ocell} \\\\")
tex += ["\\bottomrule", "\\end{tabular}"]
open(os.path.join(DIR, "rivals_table.tex"), "w").write("\n".join(tex))
print("\n--- LaTeX written to rivals_table.tex ---")
print("\n".join(tex))
