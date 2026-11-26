#!/usr/bin/env python3
"""Solve a bluff-game-v1 extensive-form game with CFR+ (alternating updates,
linear averaging) and write the average strategy in bluff-strategy-v1 format.

Usage: python3 -I solve_leduc.py [game.json] [strategy.json] [iterations]
"""
import json
import sys
from fractions import Fraction

GAME_PATH = sys.argv[1] if len(sys.argv) > 1 else "game.json"
OUT_PATH = sys.argv[2] if len(sys.argv) > 2 else "strategy.json"
ITERATIONS = int(sys.argv[3]) if len(sys.argv) > 3 else 3000


def load_game(path):
    with open(path, encoding="utf-8") as f:
        g = json.load(f)
    if g.get("format") != "bluff-game-v1":
        raise SystemExit(f"unsupported game format: {g.get('format')!r}")
    nodes = {n["id"]: n for n in g["nodes"]}
    return g, nodes, g["root"]


def build(nodes):
    """Index the tree: per-node kind/children/probs, and infoset tables."""
    info = {}          # infoset name -> index
    inf_player = []
    inf_actions = []   # list of action label tuples
    kind, children, probs, payoff, player, infoset = {}, {}, {}, {}, {}, {}
    for nid, n in nodes.items():
        if n["kind"] == "terminal":
            kind[nid] = "T"
            payoff[nid] = float(Fraction(n["payoff_p0"]))
        elif n["kind"] == "chance":
            kind[nid] = "C"
            children[nid] = [o["child"] for o in n["outcomes"]]
            probs[nid] = [float(Fraction(o["prob"])) for o in n["outcomes"]]
        elif n["kind"] == "decision":
            kind[nid] = "D"
            p = int(n["player"])
            labels = tuple(a["label"] for a in n["actions"])
            name = n["infoset"]
            if name not in info:
                info[name] = len(inf_player)
                inf_player.append(p)
                inf_actions.append(labels)
            else:
                k = info[name]
                if inf_player[k] != p or inf_actions[k] != labels:
                    raise SystemExit(f"inconsistent infoset {name!r}")
            infoset[nid] = info[name]
            player[nid] = p
            children[nid] = [a["child"] for a in n["actions"]]
        else:
            raise SystemExit(f"unknown node kind {n['kind']!r}")
    return info, inf_player, inf_actions, kind, children, probs, payoff, player, infoset


def main():
    g, nodes, root = load_game(GAME_PATH)
    (info, inf_player, inf_actions, kind, children, probs,
     payoff, player, infoset) = build(nodes)
    n_inf = len(inf_player)
    # Preorder list of nodes from root; index nodes into dense arrays for speed.
    ids = list(nodes.keys())
    idx = {nid: i for i, nid in enumerate(ids)}
    K = [kind[nid] for nid in ids]
    CH = [[idx[c] for c in children[nid]] if nid in children else None for nid in ids]
    PR = [probs.get(nid) for nid in ids]
    PAY = [payoff.get(nid, 0.0) for nid in ids]
    PL = [player.get(nid, -1) for nid in ids]
    IS = [infoset.get(nid, -1) for nid in ids]
    ROOT = idx[root]
    n_act = [len(a) for a in inf_actions]

    regret = [[0.0] * n_act[k] for k in range(n_inf)]
    strat_sum = [[0.0] * n_act[k] for k in range(n_inf)]

    def current_strategy(k):
        r = regret[k]
        pos = [x if x > 0.0 else 0.0 for x in r]
        s = sum(pos)
        if s > 0.0:
            return [x / s for x in pos]
        return [1.0 / len(r)] * len(r)

    def cfr(h, trav, t, reach0, reach1, reach_c):
        """Return expected p0 payoff from node h under current strategies.
        trav is the player whose regrets/strategy sums are updated this pass."""
        k = K[h]
        if k == "T":
            return PAY[h]
        if k == "C":
            total = 0.0
            for c, p in zip(CH[h], PR[h]):
                if p == 0.0:
                    continue
                total += p * cfr(c, trav, t, reach0, reach1, reach_c * p)
            return total
        # decision node
        pl = PL[h]
        I = IS[h]
        sigma = current_strategy(I)
        ch = CH[h]
        if pl == trav:
            vals = []
            for a, c in enumerate(ch):
                if pl == 0:
                    vals.append(cfr(c, trav, t, reach0 * sigma[a], reach1, reach_c))
                else:
                    vals.append(cfr(c, trav, t, reach0, reach1 * sigma[a], reach_c))
            v = sum(sigma[a] * vals[a] for a in range(len(ch)))
            # counterfactual reach: opponent reach times chance reach
            opp = reach_c * (reach1 if pl == 0 else reach0)
            sign = 1.0 if pl == 0 else -1.0
            r = regret[I]
            for a in range(len(ch)):
                r[a] = max(r[a] + opp * sign * (vals[a] - v), 0.0)
            own = reach0 if pl == 0 else reach1
            ss = strat_sum[I]
            w = t * own
            for a in range(len(ch)):
                ss[a] += w * sigma[a]
            return v
        # opponent decision: follow sigma, no update
        v = 0.0
        for a, c in enumerate(ch):
            if pl == 0:
                v += sigma[a] * cfr(c, trav, t, reach0 * sigma[a], reach1, reach_c)
            else:
                v += sigma[a] * cfr(c, trav, t, reach0, reach1 * sigma[a], reach_c)
        return v

    for t in range(1, ITERATIONS + 1):
        cfr(ROOT, 0, t, 1.0, 1.0, 1.0)
        cfr(ROOT, 1, t, 1.0, 1.0, 1.0)

    avg = []
    for k in range(n_inf):
        s = sum(strat_sum[k])
        if s > 0.0:
            avg.append([x / s for x in strat_sum[k]])
        else:
            avg.append([1.0 / n_act[k]] * n_act[k])

    # ---- Exact-ish evaluation: expected payoff and best responses ----------
    def avg_strategy_value(h):
        k = K[h]
        if k == "T":
            return PAY[h]
        if k == "C":
            return sum(p * avg_strategy_value(c) for c, p in zip(CH[h], PR[h]))
        sig = avg[IS[h]]
        return sum(sig[a] * avg_strategy_value(c) for a, c in enumerate(CH[h]))

    value_p0 = avg_strategy_value(ROOT)

    def best_response(br):
        """Best-response value to player (1-br)'s average strategy, measured in
        player br's payoff. Opponent/chance reach is carried down the tree;
        each br infoset picks the single action maximizing the reach-weighted sum
        over its nodes (perfect recall makes this exact)."""
        sign = 1.0 if br == 0 else -1.0
        # Opponent-plus-chance reach from root to each node.
        reach = {}
        order = []
        stack = [(ROOT, 1.0)]
        while stack:
            h, r = stack.pop()
            reach[h] = r
            order.append(h)
            k = K[h]
            if k == "C":
                for c, p in zip(CH[h], PR[h]):
                    stack.append((c, r * p))
            elif k == "D":
                if PL[h] == br:
                    for c in CH[h]:
                        stack.append((c, r))
                else:
                    sig = avg[IS[h]]
                    for a, c in enumerate(CH[h]):
                        stack.append((c, r * sig[a]))
        nodes_of = {}
        for h in order:
            if K[h] == "D" and PL[h] == br:
                nodes_of.setdefault(IS[h], []).append(h)
        choice = {}
        memo = {}

        def value(h):
            if h in memo:
                return memo[h]
            k = K[h]
            if k == "T":
                res = sign * PAY[h]
            elif k == "C":
                res = sum(p * value(c) for c, p in zip(CH[h], PR[h]))
            elif PL[h] == br:
                I = IS[h]
                if I not in choice:
                    qs = [0.0] * n_act[I]
                    for hh in nodes_of[I]:
                        for a, c in enumerate(CH[hh]):
                            qs[a] += reach[hh] * value(c)
                    choice[I] = max(range(n_act[I]), key=lambda a: qs[a])
                res = value(CH[h][choice[I]])
            else:
                sig = avg[IS[h]]
                res = sum(sig[a] * value(c) for a, c in enumerate(CH[h]))
            memo[h] = res
            return res

        return value(ROOT)

    br0 = best_response(0)   # player 0's best response value (p0 payoff)
    br1 = best_response(1)   # player 1's best response value (p1 payoff)
    exploitability = (br0 + br1) / 2.0

    # ---- Write strategy.json ----------------------------------------------
    probs_out = {}
    for name, k in info.items():
        labels = inf_actions[k]
        probs_out[name] = {lab: repr(float(avg[k][a])) for a, lab in enumerate(labels)}
    out = {
        "format": "bluff-strategy-v1",
        "game": g.get("name", "leduc"),
        "producer": (f"CFR+ (alternating, linear averaging), {ITERATIONS} iterations, "
                     f"solved by solve_leduc.py"),
        "probs": probs_out,
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
        f.write("\n")

    print(f"infosets: {n_inf}")
    print(f"iterations: {ITERATIONS}")
    print(f"value_p0 (avg strategies): {value_p0!r}")
    print(f"BR value p0 vs P1 avg: {br0!r}")
    print(f"BR value p1 vs P0 avg: {br1!r}")
    print(f"exploitability: {exploitability!r}")


if __name__ == "__main__":
    main()
