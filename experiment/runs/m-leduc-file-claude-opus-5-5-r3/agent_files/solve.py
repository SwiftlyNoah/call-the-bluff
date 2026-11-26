"""CFR+ solver for a bluff-game-v1 game tree. Reads game.json, writes strategy.json."""
import json, sys, time
from fractions import Fraction

def num(s): return float(Fraction(s))

def load(path):
    g = json.load(open(path))
    nodes = {n["id"]: n for n in g["nodes"]}
    return g, nodes

def build(g, nodes):
    # Compact arrays: kind, player, infoset index, children, probs, payoff
    infosets = {}   # name -> (idx, player, labels)
    order = []
    stack = [g["root"]]
    kind = {}; child = {}; prob = {}; pay = {}; iso = {}; plyr = {}
    while stack:
        nid = stack.pop(); n = nodes[nid]; k = n["kind"]; kind[nid] = k; order.append(nid)
        if k == "terminal":
            pay[nid] = num(n["payoff_p0"])
        elif k == "chance":
            child[nid] = [o["child"] for o in n["outcomes"]]
            prob[nid] = [num(o["prob"]) for o in n["outcomes"]]
            stack.extend(child[nid])
        else:
            labels = [a["label"] for a in n["actions"]]
            name = n["infoset"]
            if name not in infosets:
                infosets[name] = (len(infosets), n["player"], labels)
            else:
                assert infosets[name][1] == n["player"] and infosets[name][2] == labels, name
            iso[nid] = infosets[name][0]; plyr[nid] = n["player"]
            child[nid] = [a["child"] for a in n["actions"]]
            stack.extend(child[nid])
    return infosets, kind, child, prob, pay, iso, plyr

def main(iters=int(sys.argv[1]) if len(sys.argv) > 1 else 20000):
    g, nodes = load("game.json")
    infosets, kind, child, prob, pay, iso, plyr = build(g, nodes)
    root = g["root"]
    nI = len(infosets)
    nA = [0]*nI
    for name, (i, p, labels) in infosets.items(): nA[i] = len(labels)
    regret = [[0.0]*nA[i] for i in range(nI)]
    avg = [[0.0]*nA[i] for i in range(nI)]
    strat = [[1.0/nA[i]]*nA[i] for i in range(nI)]

    def cfr(nid, trav, reach_opp, reach_self, w):
        k = kind[nid]
        if k == "terminal":
            return pay[nid] if trav == 0 else -pay[nid]
        if k == "chance":
            return sum(p * cfr(c, trav, reach_opp * p, reach_self, w) for c, p in zip(child[nid], prob[nid]))
        I = iso[nid]; s = strat[I]; cs = child[nid]
        if plyr[nid] == trav:
            vals = [cfr(c, trav, reach_opp, reach_self * s[a], w) for a, c in enumerate(cs)]
            v = sum(s[a]*vals[a] for a in range(len(cs)))
            r = regret[I]
            for a in range(len(cs)): r[a] += reach_opp * (vals[a] - v)
            if w:
                ag = avg[I]
                for a in range(len(cs)): ag[a] += w * reach_self * s[a]
            return v
        else:
            return sum(s[a] * cfr(c, trav, reach_opp * s[a], reach_self, w) for a, c in enumerate(cs) if s[a] > 0)

    def update(player):
        for name, (I, p, labels) in infosets.items():
            if p != player: continue
            r = regret[I]
            for a in range(len(r)):
                if r[a] < 0: r[a] = 0.0
            tot = sum(r)
            strat[I] = [x/tot for x in r] if tot > 0 else [1.0/len(r)]*len(r)

    t0 = time.time()
    for t in range(1, iters+1):
        for trav in (0, 1):
            # avg strategy of trav accumulated during trav's own traversal (linear weights, CFR+)
            cfr(root, trav, 1.0, 1.0, float(t))
            update(trav)
        if t % 500 == 0:
            A = normalized(avg, nA)
            e, v = exploitability(root, kind, child, prob, pay, iso, plyr, infosets, A)
            print(f"iter {t} expl {e:.3e} v0 {v:.6f} ({time.time()-t0:.1f}s)", flush=True)
            if e < 1e-6: break

    A = normalized(avg, nA)
    e, v = exploitability(root, kind, child, prob, pay, iso, plyr, infosets, A)
    print(f"final expl {e:.6e} value_p0 {v:.8f}")
    probs = {}
    for name, (I, p, labels) in infosets.items():
        probs[name] = {lab: repr(A[I][a]) for a, lab in enumerate(labels)}
    json.dump({"format": "bluff-strategy-v1", "game": g.get("name", "leduc"),
               "producer": f"CFR+ (alternating, linear averaging), {t} iterations; exploitability {e:.3e}",
               "probs": probs}, open("strategy.json", "w"), indent=1)

def normalized(avg, nA):
    out = []
    for i, a in enumerate(avg):
        s = sum(a)
        out.append([x/s for x in a] if s > 0 else [1.0/nA[i]]*nA[i])
    return out

def exploitability(root, kind, child, prob, pay, iso, plyr, infosets, A):
    # value of profile
    def ev(nid):
        k = kind[nid]
        if k == "terminal": return pay[nid]
        if k == "chance": return sum(p*ev(c) for c, p in zip(child[nid], prob[nid]))
        s = A[iso[nid]]
        return sum(s[a]*ev(c) for a, c in enumerate(child[nid]) if s[a] > 0)
    v0 = ev(root)
    brs = []
    for br in (0, 1):
        sign = 1.0 if br == 0 else -1.0
        # collect br player's nodes per infoset with opponent*chance reach, and depth
        members = {}; depth = {}
        def collect(nid, reach, d):
            k = kind[nid]
            if k == "terminal": return
            if k == "chance":
                for c, p in zip(child[nid], prob[nid]): collect(c, reach*p, d+1)
                return
            I = iso[nid]
            if plyr[nid] == br:
                members.setdefault(I, []).append((nid, reach)); depth[I] = max(depth.get(I, 0), d)
                for c in child[nid]: collect(c, reach, d+1)
            else:
                s = A[I]
                for a, c in enumerate(child[nid]): collect(c, reach*s[a], d+1)
        collect(root, 1.0, 0)
        choice = {}
        def val(nid):  # br player's value (unweighted by reach)
            k = kind[nid]
            if k == "terminal": return sign*pay[nid]
            if k == "chance": return sum(p*val(c) for c, p in zip(child[nid], prob[nid]))
            I = iso[nid]
            if plyr[nid] == br: return val(child[nid][choice[I]])
            s = A[I]
            return sum(s[a]*val(c) for a, c in enumerate(child[nid]) if s[a] > 0)
        for I in sorted(members, key=lambda I: -depth[I]):
            n_a = len(child[members[I][0][0]])
            q = [sum(r*val(child[nid][a]) for nid, r in members[I]) for a in range(n_a)]
            choice[I] = max(range(n_a), key=lambda a: q[a])
        brs.append(val(root))
    return (brs[0] + brs[1]) / 2, v0

if __name__ == "__main__":
    main()
