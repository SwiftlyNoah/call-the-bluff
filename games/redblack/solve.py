"""An equilibrium certificate for a small game file, independent of reference/.

    uv run games/redblack/solve.py games/out/redblack/rb-pennies.game.json [--iters 4000]

1. CFR+ (floats, alternating updates, linear averaging) for the average profile.
2. Round every row to small rationals (Fraction.limit_denominator(d), d = 1,
   2, ..., 64) and compute, in exact Fraction arithmetic, both players'
   best-response values against the rounded profile.
3. If BR_0 + BR_1 == 0 exactly, the rounded profile is an exact equilibrium
   and its value is the game value (minimax): that is the certificate. The
   rounded profile is written as a strategy file (bluff-strategy-v1) next to
   the game, so the Dafny checker can confirm exploitability 0 itself.

Nothing is reported as exact unless the exact NashConv is 0.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

sys.setrecursionlimit(10000)


def load(path: Path) -> dict:
    doc = json.loads(path.read_text())
    nodes = {n["id"]: n for n in doc["nodes"]}
    isets: dict[str, dict] = {}
    for n in doc["nodes"]:
        if n["kind"] == "decision":
            isets.setdefault(n["infoset"], {"player": n["player"], "actions": [a["label"] for a in n["actions"]], "nodes": []})
            isets[n["infoset"]]["nodes"].append(n["id"])
        elif n["kind"] == "chance":
            n["_p"] = [Fraction(o["prob"]) for o in n["outcomes"]]
        else:
            n["_u"] = Fraction(n["payoff_p0"])
    return {"doc": doc, "nodes": nodes, "isets": isets, "root": doc["root"]}


# ---------------------------------------------------------------------------
# Exact best response


def best_response(g: dict, sigma: dict, p: int) -> tuple[Fraction, dict]:
    """Max over pure strategies of player p of p's expected utility against
    sigma (only the opponent's rows are read). Perfect recall lets infosets be
    decided deepest own-history first."""
    nodes, root = g["nodes"], g["root"]
    reach: dict[int, Fraction] = {}
    depth: dict[str, int] = {}
    stack = [(root, Fraction(1), 0)]
    while stack:
        nid, r, d = stack.pop()
        reach[nid] = r
        n = nodes[nid]
        if n["kind"] == "chance":
            for o, pr in zip(n["outcomes"], n["_p"]):
                stack.append((o["child"], r * pr, d))
        elif n["kind"] == "decision":
            if n["player"] == p:
                depth[n["infoset"]] = d
                for a in n["actions"]:
                    stack.append((a["child"], r, d + 1))
            else:
                row = sigma[n["infoset"]]
                for a in n["actions"]:
                    stack.append((a["child"], r * row[a["label"]], d))
    choice: dict[str, str] = {}
    memo: dict[int, Fraction] = {}
    sign = 1 if p == 0 else -1

    def val(nid: int) -> Fraction:
        if nid in memo:
            return memo[nid]
        n = nodes[nid]
        if n["kind"] == "terminal":
            v = sign * n["_u"]
        elif n["kind"] == "chance":
            v = sum((pr * val(o["child"]) for o, pr in zip(n["outcomes"], n["_p"])), Fraction(0))
        elif n["player"] == p:
            lab = choice[n["infoset"]]
            v = val(next(a["child"] for a in n["actions"] if a["label"] == lab))
        else:
            row = sigma[n["infoset"]]
            v = sum((row[a["label"]] * val(a["child"]) for a in n["actions"] if row[a["label"]]), Fraction(0))
        memo[nid] = v
        return v

    for iset in sorted(depth, key=lambda k: -depth[k]):
        info = g["isets"][iset]
        best, best_lab = None, None
        for k, lab in enumerate(info["actions"]):
            tot = sum((reach[nid] * val(nodes[nid]["actions"][k]["child"]) for nid in info["nodes"]), Fraction(0))
            if best is None or tot > best:
                best, best_lab = tot, lab
        choice[iset] = best_lab
    return val(root), choice


def profile_value(g: dict, sigma: dict) -> Fraction:
    nodes = g["nodes"]

    def val(nid: int) -> Fraction:
        n = nodes[nid]
        if n["kind"] == "terminal":
            return n["_u"]
        if n["kind"] == "chance":
            return sum((pr * val(o["child"]) for o, pr in zip(n["outcomes"], n["_p"])), Fraction(0))
        row = sigma[n["infoset"]]
        return sum((row[a["label"]] * val(a["child"]) for a in n["actions"] if row[a["label"]]), Fraction(0))

    return val(g["root"])


# ---------------------------------------------------------------------------
# CFR+


def cfr_plus(g: dict, iters: int) -> dict:
    nodes, isets = g["nodes"], g["isets"]
    regret = {k: [0.0] * len(v["actions"]) for k, v in isets.items()}
    avg = {k: [0.0] * len(v["actions"]) for k, v in isets.items()}

    def strat(k: str) -> list[float]:
        r = regret[k]
        s = sum(r)
        return [x / s for x in r] if s > 0 else [1.0 / len(r)] * len(r)

    # Lesson learned here (see README "A bug in our own CFR+"): never prune an
    # opponent branch whose current probability is 0. The average strategy is
    # weighted by OWN reach only, so it must still be accumulated below such a
    # branch; pruning left rows at their iteration-1 values and an average
    # profile exploitable by 0.75 while every current strategy was fine.
    delta: dict[str, list[float]] = {}
    cur: dict[str, list[float]] = {}

    def walk(nid: int, upd: int, r_opp: float, r_own: float, t: int) -> float:
        n = nodes[nid]
        if n["kind"] == "terminal":
            return float(n["_u"]) * (1 if upd == 0 else -1)
        if n["kind"] == "chance":
            return sum(float(pr) * walk(o["child"], upd, r_opp * float(pr), r_own, t) for o, pr in zip(n["outcomes"], n["_p"]))
        k = n["infoset"]
        s = cur.get(k)
        if s is None:
            s = cur[k] = strat(k)  # fixed for the whole traversal
        if n["player"] == upd:
            vals = [walk(a["child"], upd, r_opp, r_own * s[i], t) for i, a in enumerate(n["actions"])]
            v = sum(si * vi for si, vi in zip(s, vals))
            d = delta.setdefault(k, [0.0] * len(vals))
            for i in range(len(vals)):
                d[i] += r_opp * (vals[i] - v)
            av = avg[k]
            for i in range(len(s)):
                av[i] += t * r_own * s[i]
            return v
        return sum(s[i] * walk(a["child"], upd, r_opp * s[i], r_own, t) for i, a in enumerate(n["actions"]))

    for t in range(1, iters + 1):
        for upd in (0, 1):
            delta.clear()
            cur.clear()
            walk(g["root"], upd, 1.0, 1.0, t)
            for k, d in delta.items():
                regret[k] = [max(0.0, r + x) for r, x in zip(regret[k], d)]
    out = {}
    for k, v in isets.items():
        tot = sum(avg[k])
        out[k] = [x / tot for x in avg[k]] if tot > 0 else [1.0 / len(v["actions"])] * len(v["actions"])
    return out


def rounded(g: dict, avg: dict, d: int) -> dict:
    sigma = {}
    for k, v in g["isets"].items():
        row = [Fraction(x).limit_denominator(d) for x in avg[k]]
        s = sum(row)
        if s == 0:
            row, s = [Fraction(1)] * len(row), Fraction(len(row))
        sigma[k] = {lab: x / s for lab, x in zip(v["actions"], row)}
    return sigma


def q(f: Fraction) -> str:
    return str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("game")
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--max-den", type=int, default=64)
    args = ap.parse_args(argv)
    path = Path(args.game)
    g = load(path)
    avg = cfr_plus(g, args.iters)
    best = None
    for d in range(1, args.max_den + 1):
        sigma = rounded(g, avg, d)
        b0, _ = best_response(g, sigma, 0)
        b1, _ = best_response(g, sigma, 1)
        conv = b0 + b1
        if best is None or conv < best[0]:
            best = (conv, d, sigma, b0, b1)
        if conv == 0:
            break
    conv, d, sigma, b0, b1 = best
    v = profile_value(g, sigma)
    res = {
        "game": g["doc"]["name"],
        "cfr_plus_iters": args.iters,
        "denominator_bound": d,
        "br_value_p0": q(b0),
        "br_value_p1": q(b1),
        "nash_conv": q(conv),
        "certified_exact": conv == 0,
        "value_p0": q(v),
        "p0_win_probability": q((1 + v) / 2),
    }
    print(json.dumps(res))
    if conv == 0:
        out = path.with_name(path.name.replace(".game.json", ".cfr.strategy.json"))
        out.write_text(json.dumps({
            "format": "bluff-strategy-v1",
            "game": g["doc"]["name"],
            "producer": f"games/redblack/solve.py CFR+ {args.iters} iters, rounded limit_denominator({d}), exact NashConv 0",
            "probs": {k: {lab: q(x) for lab, x in row.items()} for k, row in sigma.items()},
        }, indent=1) + "\n")
    return 0 if conv == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
