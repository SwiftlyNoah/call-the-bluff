#!/usr/bin/env python3
"""CFR+ solver for a bluff-game-v1 game tree. Reads game.json, writes strategy.json."""
import json
import sys
from fractions import Fraction

ITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 20000


def num(s):
    return float(Fraction(s))


def load(path):
    g = json.load(open(path))
    nodes = {n["id"]: n for n in g["nodes"]}
    return g, nodes


def build(g, nodes):
    # Flatten into arrays in preorder for fast iteration.
    order = []
    stack = [g["root"]]
    while stack:
        i = stack.pop()
        order.append(i)
        n = nodes[i]
        if n["kind"] == "chance":
            stack.extend(o["child"] for o in reversed(n["outcomes"]))
        elif n["kind"] == "decision":
            stack.extend(a["child"] for a in reversed(n["actions"]))
    infosets = {}
    for i in order:
        n = nodes[i]
        if n["kind"] == "decision":
            labels = [a["label"] for a in n["actions"]]
            if n["infoset"] in infosets:
                p, l = infosets[n["infoset"]]
                assert p == n["player"] and l == labels, n["infoset"]
            else:
                infosets[n["infoset"]] = (n["player"], labels)
    return order, infosets


def main():
    g, nodes = load("game.json")
    order, infosets = build(g, nodes)
    names = list(infosets)
    idx = {k: j for j, k in enumerate(names)}
    nact = [len(infosets[k][1]) for k in names]
    regret = [[0.0] * m for m in nact]
    avg = [[0.0] * m for m in nact]

    # compact node representation
    kind, kids, probs, pay, player, iset = {}, {}, {}, {}, {}, {}
    for i in order:
        n = nodes[i]
        kind[i] = n["kind"]
        if n["kind"] == "chance":
            kids[i] = [o["child"] for o in n["outcomes"]]
            probs[i] = [num(o["prob"]) for o in n["outcomes"]]
        elif n["kind"] == "decision":
            kids[i] = [a["child"] for a in n["actions"]]
            player[i] = n["player"]
            iset[i] = idx[n["infoset"]]
        else:
            pay[i] = num(n["payoff_p0"])

    def current(j):
        r = regret[j]
        s = sum(x for x in r if x > 0)
        if s > 0:
            return [x / s if x > 0 else 0.0 for x in r]
        return [1.0 / len(r)] * len(r)

    def cfr(i, upd, rp, ro, t, sigma, accum):
        """Returns value for player upd. rp = upd's reach, ro = opponent*chance reach."""
        k = kind[i]
        if k == "terminal":
            return pay[i] if upd == 0 else -pay[i]
        if k == "chance":
            return sum(p * cfr(c, upd, rp, ro * p, t, sigma, accum)
                       for c, p in zip(kids[i], probs[i]))
        j = iset[i]
        s = sigma[j]
        cs = kids[i]
        if player[i] == upd:
            vals = [cfr(c, upd, rp * s[a], ro, t, sigma, accum) for a, c in enumerate(cs)]
            v = sum(s[a] * vals[a] for a in range(len(cs)))
            acc = accum[j]
            for a in range(len(cs)):
                acc[a] += ro * (vals[a] - v)
            return v
        else:
            return sum(s[a] * cfr(c, upd, rp, ro * s[a], t, sigma, accum)
                       for a, c in enumerate(cs) if s[a] > 0)


    def accum_avg(i, reach, t, sigma, pl):
        k = kind[i]
        if k == "terminal":
            return
        if k == "chance":
            for c in kids[i]:
                accum_avg(c, reach, t, sigma, pl)
            return
        j = iset[i]
        s = sigma[j]
        if player[i] == pl:
            if reach > 0:
                av = avg[j]
                for a in range(len(s)):
                    av[a] += t * reach * s[a]
            for a, c in enumerate(kids[i]):
                accum_avg(c, reach * s[a], t, sigma, pl)
        else:
            for c in kids[i]:
                accum_avg(c, reach, t, sigma, pl)

    root = g["root"]
    sys.setrecursionlimit(10000)
    for t in range(1, ITERS + 1):
        for upd in (0, 1):
            sigma = [current(j) for j in range(len(names))]
            accum = [[0.0] * m for m in nact]
            cfr(root, upd, 1.0, 1.0, t, sigma, accum)
            for j in range(len(names)):
                if infosets[names[j]][0] == upd:
                    r = regret[j]
                    acc = accum[j]
                    for a in range(len(r)):
                        r[a] = max(r[a] + acc[a], 0.0)  # CFR+
            # average strategy of upd using the strategy it played this iteration
            accum_avg(root, 1.0, t, sigma, upd)
        if t % 1000 == 0:
            strat = average(avg)
            e, v = exploitability(root, kind, kids, probs, pay, player, iset, strat)
            print(f"iter {t} exploitability {e:.3e} value_p0 {v:.6f}", flush=True)

    strat = average(avg)
    e, v = exploitability(root, kind, kids, probs, pay, player, iset, strat)
    print(f"final exploitability {e:.6e} value_p0 {v:.8f}")
    out = {
        "format": "bluff-strategy-v1",
        "game": g["name"],
        "producer": f"CFR+ (alternating updates, linear averaging), {ITERS} iterations",
        "probs": {names[j]: {lab: repr(p) for lab, p in zip(infosets[names[j]][1], strat[j])}
                  for j in range(len(names))},
    }
    json.dump(out, open("strategy.json", "w"), indent=1)
    json.dump({"exploitability": e, "value_p0": v, "iters": ITERS}, open("result_stats.json", "w"))


def average(avg):
    out = []
    for a in avg:
        s = sum(a)
        if s > 0:
            p = [x / s for x in a]
        else:
            p = [1.0 / len(a)] * len(a)
        # renormalize to guard float drift
        z = sum(p)
        out.append([x / z for x in p])
    return out


def exploitability(root, kind, kids, probs, pay, player, iset, strat):
    def ev(i):
        k = kind[i]
        if k == "terminal":
            return pay[i]
        if k == "chance":
            return sum(p * ev(c) for c, p in zip(kids[i], probs[i]))
        s = strat[iset[i]]
        return sum(s[a] * ev(c) for a, c in enumerate(kids[i]) if s[a] > 0)

    def br(pl):
        # collect nodes of each pl-infoset with their opponent*chance reach
        members = {}

        def collect(i, r):
            k = kind[i]
            if k == "terminal":
                return
            if k == "chance":
                for c, p in zip(kids[i], probs[i]):
                    collect(c, r * p)
                return
            if player[i] == pl:
                members.setdefault(iset[i], []).append((i, r))
                for c in kids[i]:
                    collect(c, r)
            else:
                s = strat[iset[i]]
                for a, c in enumerate(kids[i]):
                    collect(c, r * s[a])

        collect(root, 1.0)
        choice = {}
        memo = {}

        def val(i):  # value to pl at node i given BR choices
            if i in memo:
                return memo[i]
            k = kind[i]
            if k == "terminal":
                v = pay[i] if pl == 0 else -pay[i]
            elif k == "chance":
                v = sum(p * val(c) for c, p in zip(kids[i], probs[i]))
            elif player[i] == pl:
                j = iset[i]
                if j not in choice:
                    ms = members[j]
                    na = len(kids[i])
                    tot = [sum(r * val(kids[h][a]) for h, r in ms) for a in range(na)]
                    choice[j] = max(range(na), key=lambda a: tot[a])
                v = val(kids[i][choice[j]])
            else:
                s = strat[iset[i]]
                v = sum(s[a] * val(c) for a, c in enumerate(kids[i]) if s[a] > 0)
            memo[i] = v
            return v

        return val(root)

    b0 = br(0)
    b1 = br(1)
    return (b0 + b1) / 2, ev(root)


if __name__ == "__main__":
    main()
