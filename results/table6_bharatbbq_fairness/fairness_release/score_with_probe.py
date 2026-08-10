#!/usr/bin/env python3
"""Guard-scoring stage: apply the released discrimination probe to an e5 embedding.
Demonstrates the deterministic mapping from embedding -> discrimination score -> flag.
(Embeddings are produced by intfloat/multilingual-e5-base; see ENVIRONMENT.txt.)"""
import json, os, math, sys

HERE = os.path.dirname(os.path.abspath(__file__))
probe = json.load(open(os.path.join(HERE, "discrimination_probe.json")))
COEF, B, THR = probe["coef"], probe["intercept"], probe["decision_threshold"]

def score(embedding):
    z = B + sum(c * e for c, e in zip(COEF, embedding))
    return 1.0 / (1.0 + math.exp(-z))

def flag(embedding):
    return score(embedding) >= THR

if __name__ == "__main__":
    # demo: expects a JSON list of 768 floats on stdin (an e5 embedding)
    emb = json.load(sys.stdin)
    s = score(emb)
    print(f"discrimination_prob={s:.4f}  threshold={THR}  flag={'YES' if s>=THR else 'no'}")
