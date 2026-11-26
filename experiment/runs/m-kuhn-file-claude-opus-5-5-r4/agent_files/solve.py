"""CFR+ solver for games in bluff-game-v1 format. Usage: python3 solve.py [game.json] [strategy.json] [iters]"""
import json, sys
from fractions import Fraction
from functools import lru_cache

def num(s): return float(Fraction(s))

def load(path):
    g = json.load(open(path))
    nodes = {n["id"]: n for n in g["nodes"]}
    infosets = {}  # name -> (player, [labels], [node ids])
    for n in g["nodes"]:
        if n["kind"] == "decision":
            labels = [a["label"] for a in n["actions"]]
            if n["infoset"] not in infosets:
                infosets[n["infoset"]] = (n["player"], labels, [])
            p, l, ids = infosets[n["infoset"]]
            assert p == n["player"] and l == labels, n["infoset"]
            ids.append(n["id"])
    return g, nodes, infosets

def solve(nodes, root, infosets, iters):
    regret = {I: [0.0] * len(v[1]) for I, v in infosets.items()}
    avg = {I: [0.0] * len(v[1]) for I, v in infosets.items()}

    def current(I):
        r = [max(x, 0.0) for x in regret[I]]
        s = sum(r)
        return [x / s for x in r] if s > 0 else [1.0 / len(r)] * len(r)

    def cfr(nid, i, pi, po, t):
        """Returns value for player i. pi: player i reach, po: opponent*chance reach."""
        n = nodes[nid]
        if n["kind"] == "terminal":
            v = num(n["payoff_p0"]); return v if i == 0 else -v
        if n["kind"] == "chance":
            return sum(num(o["prob"]) * cfr(o["child"], i, pi, po * num(o["prob"]), t) for o in n["outcomes"])
        I = n["infoset"]; sig = current(I); acts = n["actions"]
        if n["player"] == i:
            vals = [cfr(a["child"], i, pi * s, po, t) for a, s in zip(acts, sig)]
            v = sum(s * x for s, x in zip(sig, vals))
            for k in range(len(acts)):
                regret[I][k] += po * (vals[k] - v)
            return v
        else:
            vals = [cfr(a["child"], i, pi, po * s, t) for a, s in zip(acts, sig)]
            # average strategy of opponent weighted linearly (CFR+), using its own reach
            return sum(s * x for s, x in zip(sig, vals))

    # strategy averaging needs each player's own reach; do a separate pass
    def accumulate(nid, reach, w):
        n = nodes[nid]
        if n["kind"] == "terminal": return
        if n["kind"] == "chance":
            for o in n["outcomes"]: accumulate(o["child"], reach, w)
            return
        I = n["infoset"]; sig = current(I); p = n["player"]
        for k, a in enumerate(n["actions"]):
            avg[I][k] += w * reach[p] * sig[k]
            r = list(reach); r[p] *= sig[k]
            accumulate(a["child"], r, w)

    for t in range(1, iters + 1):
        for i in (0, 1):  # alternating updates
            cfr(root, i, 1.0, 1.0, t)
            for I in regret:
                if infosets[I][0] == i:
                    regret[I] = [max(x, 0.0) for x in regret[I]]
        accumulate(root, [1.0, 1.0], float(t))
    strat = {}
    for I, a in avg.items():
        s = sum(a)
        strat[I] = [x / s for x in a] if s > 0 else [1.0 / len(a)] * len(a)
    return strat

def evaluate(nodes, root, infosets, strat):
    """Exact (Fraction) value for p0 and best-response values for both players."""
    S = {I: [Fraction(x) for x in v] for I, v in strat.items()}

    def value(nid):
        n = nodes[nid]
        if n["kind"] == "terminal": return Fraction(n["payoff_p0"])
        if n["kind"] == "chance":
            return sum(Fraction(o["prob"]) * value(o["child"]) for o in n["outcomes"])
        return sum(s * value(a["child"]) for s, a in zip(S[n["infoset"]], n["actions"]))

    def br(i):
        # opponent/chance reach of every node
        reach = {}
        def walk(nid, r):
            reach[nid] = r; n = nodes[nid]
            if n["kind"] == "chance":
                for o in n["outcomes"]: walk(o["child"], r * Fraction(o["prob"]))
            elif n["kind"] == "decision":
                for s, a in zip(S[n["infoset"]], n["actions"]):
                    walk(a["child"], r * s if n["player"] != i else r)
        walk(root, Fraction(1))
        choice = {}
        def v(nid):  # value to player i, unnormalized-free (per node)
            n = nodes[nid]
            if n["kind"] == "terminal":
                x = Fraction(n["payoff_p0"]); return x if i == 0 else -x
            if n["kind"] == "chance":
                return sum(Fraction(o["prob"]) * v(o["child"]) for o in n["outcomes"])
            if n["player"] != i:
                return sum(s * v(a["child"]) for s, a in zip(S[n["infoset"]], n["actions"]))
            return v(n["actions"][act(n["infoset"])]["child"])
        def act(I):
            if I not in choice:
                ids = infosets[I][2]
                tot = [sum(reach[h] * v(nodes[h]["actions"][k]["child"]) for h in ids)
                       for k in range(len(infosets[I][1]))]
                choice[I] = max(range(len(tot)), key=lambda k: tot[k])
            return choice[I]
        return v(root)

    return value(root), br(0), br(1)

def main():
    gpath = sys.argv[1] if len(sys.argv) > 1 else "game.json"
    opath = sys.argv[2] if len(sys.argv) > 2 else "strategy.json"
    iters = int(sys.argv[3]) if len(sys.argv) > 3 else 20000
    g, nodes, infosets = load(gpath)
    strat = solve(nodes, g["root"], infosets, iters)
    v0, br0, br1 = evaluate(nodes, g["root"], infosets, strat)
    expl = (br0 + br1) / 2
    print(f"iters={iters} value_p0={float(v0):.6f} br_p0={float(br0):.6f} br_p1={float(br1):.6f} exploitability={float(expl):.3e}")
    out = {"format": "bluff-strategy-v1", "game": g.get("name", "kuhn"),
           "producer": f"solve.py: CFR+ (alternating updates, linear averaging), {iters} iterations",
           "probs": {}}
    for I, (p, labels, _) in sorted(infosets.items()):
        probs = strat[I]
        strs = [repr(x) for x in probs[:-1]]
        last = Fraction(1) - sum(Fraction(s) for s in strs)  # exact sum to 1
        strs.append(repr(float(last)) if Fraction(float(last)) == last else str(last))
        out["probs"][I] = dict(zip(labels, strs))
    json.dump(out, open(opath, "w"), indent=1)
    for I, d in out["probs"].items(): print(I, d)

if __name__ == "__main__":
    main()
