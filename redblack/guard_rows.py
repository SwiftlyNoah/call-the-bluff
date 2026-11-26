#!/usr/bin/env python3
"""Which guarded rows move a class's exploitability? (review-redblack-stretch)

    uv run redblack/extract.py d0-3.2.1-1.0.1 --guards --no-br
    uv run redblack/guard_rows.py d0-3.2.1-1.0.1

Starts from the served profile and swaps in the guarded version of one group
of rows at a time (grouped by seat and phase, then seat-0 bidding rows by
situation), printing (exploitability, gain_p0, gain_p1) from extract.py's
exact Fraction best response.
"""
import json, sys
from fractions import Fraction
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import extract as X
c = sys.argv[1]
g = json.load(open(X.OUT / "games" / f"{c}.game.json"))["nodes"]
def load(l):
    return {I: {a: Fraction(p) for a, p in r.items()} for I, r in json.load(open(X.OUT / "strategies" / f"{c}.{l}.strategy.json"))["probs"].items()}
S, G = load("served"), load("guarded")
def ex(t):
    v, b0, b1, *_ = X.exact_eval(g, 0, t); return float((b0 + b1) / 2), float(b0 - v), float(b1 + v)
print("served", ex(S), "guarded", ex(G))
diff = [k for k in S if S[k] != G[k]]
groups = {}
for k in diff:
    groups.setdefault(k.split("|")[0] + "|" + k.split("|")[1], []).append(k)
for name, ks in groups.items():
    t = dict(S); 
    for k in ks: t[k] = G[k]
    print("only", name, len(ks), "guarded rows ->", ex(t))
sub = {}
for k in groups.get("p0|B", []):
    sub.setdefault("|".join(k.split("|")[1:5]), []).append(k)
for name, ks in sub.items():
    t = dict(S)
    for k in ks: t[k] = G[k]
    print("p0 sub", name, len(ks), ex(t), S[ks[0]], G[ks[0]])
