#!/usr/bin/env python3
"""CFR+ solver for bluff-game-v1 games (two-player, zero-sum, imperfect information).

Usage: python3 -I solve_cfr.py [game.json] [strategy.json] [iterations]

Reads the game tree from game.json, runs simultaneous CFR+ with linear averaging,
computes the exploitability of the average strategy exactly (best responses over the
tree, respecting infosets), and writes strategy.json in bluff-strategy-v1 format.
"""
import json
import os
import sys
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_num(s):
    return float(Fraction(s))


def load_game(path):
    with open(path) as f:
        g = json.load(f)
    assert g["format"] == "bluff-game-v1", g.get("format")
    nodes = {n["id"]: n for n in g["nodes"]}
    # Infoset table: name -> index, with player and action labels.
    infoset_index, infoset_player, infoset_actions = {}, [], []
    # Compact node representation.
    kind, player, iset, children, probs, payoff = {}, {}, {}, {}, {}, {}
    for n in g["nodes"]:
        nid = n["id"]
        if n["kind"] == "terminal":
            kind[nid] = "T"
            payoff[nid] = parse_num(n["payoff_p0"])
        elif n["kind"] == "chance":
            kind[nid] = "C"
            children[nid] = [o["child"] for o in n["outcomes"]]
            probs[nid] = [parse_num(o["prob"]) for o in n["outcomes"]]
            s = sum(probs[nid])
            assert abs(s - 1.0) < 1e-12, f"chance node {nid} probs sum to {s}"
        elif n["kind"] == "decision":
            kind[nid] = "D"
            p = int(n["player"])
            labels = [a["label"] for a in n["actions"]]
            name = n["infoset"]
            if name not in infoset_index:
                infoset_index[name] = len(infoset_player)
                infoset_player.append(p)
                infoset_actions.append(labels)
            else:
                i = infoset_index[name]
                assert infoset_player[i] == p and infoset_actions[i] == labels, name
            player[nid] = p
            iset[nid] = infoset_index[name]
            children[nid] = [a["child"] for a in n["actions"]]
        else:
            raise ValueError(n["kind"])
    return {
        "name": g["name"],
        "root": g["root"],
        "nodes": nodes,
        "kind": kind,
        "player": player,
        "iset": iset,
        "children": children,
        "probs": probs,
        "payoff": payoff,
        "iset_names": list(infoset_index.keys()),
        "iset_player": infoset_player,
        "iset_actions": infoset_actions,
    }


def bfs_order(G):
    order, seen, frontier = [], set(), [G["root"]]
    while frontier:
        nxt = []
        for n in frontier:
            if n in seen:
                continue
            seen.add(n)
            order.append(n)
            nxt.extend(G["children"].get(n, []))
        frontier = nxt
    return order


def regret_match(r):
    pos = [max(x, 0.0) for x in r]
    s = sum(pos)
    if s > 0:
        return [x / s for x in pos]
    return [1.0 / len(r)] * len(r)


def cfr_plus(G, iterations, log_every=1000):
    n_is = len(G["iset_player"])
    R = [[0.0] * len(G["iset_actions"][i]) for i in range(n_is)]
    S = [[0.0] * len(G["iset_actions"][i]) for i in range(n_is)]
    kind, player, iset, children, probs, payoff = (
        G["kind"], G["player"], G["iset"], G["children"], G["probs"], G["payoff"])
    sigma_cache = None
    history = []

    def traverse(n, p0r, p1r, pcr, t, sig, dR, dS):
        k = kind[n]
        if k == "T":
            return payoff[n]
        if k == "C":
            v = 0.0
            for c, pr in zip(children[n], probs[n]):
                v += pr * traverse(c, p0r, p1r, pcr * pr, t, sig, dR, dS)
            return v
        p = player[n]
        I = iset[n]
        s = sig[I]
        own = p0r if p == 0 else p1r
        opp = (p1r if p == 0 else p0r) * pcr
        cvals = []
        for a, c in enumerate(children[n]):
            if p == 0:
                cvals.append(traverse(c, p0r * s[a], p1r, pcr, t, sig, dR, dS))
            else:
                cvals.append(traverse(c, p0r, p1r * s[a], pcr, t, sig, dR, dS))
        v = sum(s[a] * cvals[a] for a in range(len(s)))
        # Regret for player p's actions, weighted by opponent+chance reach.
        sign = 1.0 if p == 0 else -1.0
        for a in range(len(s)):
            dR[I][a] += opp * sign * (cvals[a] - v)
        # Average strategy weighted by own reach and linear iteration weight t.
        for a in range(len(s)):
            dS[I][a] += t * own * s[a]
        return v

    for t in range(1, iterations + 1):
        sig = [regret_match(R[i]) for i in range(n_is)]
        dR = [[0.0] * len(R[i]) for i in range(n_is)]
        dS = [[0.0] * len(S[i]) for i in range(n_is)]
        traverse(G["root"], 1.0, 1.0, 1.0, t, sig, dR, dS)
        for i in range(n_is):
            for a in range(len(R[i])):
                R[i][a] = max(0.0, R[i][a] + dR[i][a])
                S[i][a] += dS[i][a]
        if log_every and (t % log_every == 0 or t == iterations):
            avg = average_strategy(G, S)
            e = exploitability(G, avg)
            print(f"iter {t}: exploitability {e:.3e}", file=sys.stderr, flush=True)
            history.append((t, e))
    return average_strategy(G, S), history


def average_strategy(G, S):
    avg = []
    for i, row in enumerate(S):
        tot = sum(row)
        if tot > 0:
            avg.append([x / tot for x in row])
        else:
            avg.append([1.0 / len(row)] * len(row))
    return avg


def best_response_value(G, strat, br):
    """Value (player 0's payoff) when player `br` best-responds to `strat`.

    br=0 maximizes player 0's payoff; br=1 minimizes it (player 1 maximizes its own).
    The other player follows strat. Choices are made per infoset, using reach
    probabilities from chance and the fixed opponent only (perfect recall makes this exact).
    """
    kind, player, iset, children, probs, payoff = (
        G["kind"], G["player"], G["iset"], G["children"], G["probs"], G["payoff"])
    order = bfs_order(G)
    w = {G["root"]: 1.0}
    for n in order:
        if n not in w:
            continue
        k = kind[n]
        if k == "C":
            for c, pr in zip(children[n], probs[n]):
                w[c] = w.get(c, 0.0) + w[n] * pr
        elif k == "D":
            if player[n] == br:
                for c in children[n]:
                    w[c] = w.get(c, 0.0) + w[n]
            else:
                s = strat[iset[n]]
                for a, c in enumerate(children[n]):
                    w[c] = w.get(c, 0.0) + w[n] * s[a]
    memo_val, choice = {}, {}

    def val(n):
        if n in memo_val:
            return memo_val[n]
        k = kind[n]
        if k == "T":
            r = payoff[n]
        elif k == "C":
            r = sum(pr * val(c) for c, pr in zip(children[n], probs[n]))
        elif player[n] == br:
            I = iset[n]
            if I not in choice:
                kids = children[n]
                # Gather all nodes of this infoset that are reachable.
                nodes_I = G["infoset_nodes"][I]
                best_a, best_v = None, None
                for a in range(len(G["iset_actions"][I])):
                    tot = 0.0
                    for m in nodes_I:
                        tot += w.get(m, 0.0) * val(children[m][a])
                    if best_v is None or (tot > best_v if br == 0 else tot < best_v):
                        best_a, best_v = a, tot
                choice[I] = best_a
            r = val(children[n][choice[I]])
        else:
            s = strat[iset[n]]
            r = sum(s[a] * val(c) for a, c in enumerate(children[n]))
        memo_val[n] = r
        return r

    return val(G["root"])


def exploitability(G, strat):
    br0 = best_response_value(G, strat, 0)  # player 0 best response to strat
    br1 = -best_response_value(G, strat, 1)  # player 1 best response (its payoff)
    return (br0 + br1) / 2.0


def game_value(G, strat):
    """Expected payoff to player 0 when both players follow strat."""
    kind, player, iset, children, probs, payoff = (
        G["kind"], G["player"], G["iset"], G["children"], G["probs"], G["payoff"])
    memo = {}

    def val(n):
        if n in memo:
            return memo[n]
        k = kind[n]
        if k == "T":
            r = payoff[n]
        elif k == "C":
            r = sum(pr * val(c) for c, pr in zip(children[n], probs[n]))
        else:
            s = strat[iset[n]]
            r = sum(s[a] * val(c) for a, c in enumerate(children[n]))
        memo[n] = r
        return r

    return val(G["root"])


def write_strategy(G, avg, path, producer):
    probs = {}
    for i, name in enumerate(G["iset_names"]):
        labels = G["iset_actions"][i]
        tot = sum(avg[i])
        probs[name] = {lab: repr(float(x / tot)) for lab, x in zip(labels, avg[i])}
    out = {
        "format": "bluff-strategy-v1",
        "game": G["name"],
        "producer": producer,
        "probs": probs,
    }
    with open(path, "w") as f:
        json.dump(out, f, indent=1)


def main():
    game_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "game.json")
    out_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "strategy.json")
    iters = int(sys.argv[3]) if len(sys.argv) > 3 else 20000
    G = load_game(game_path)
    # Infoset -> list of node ids (for best responses).
    G["infoset_nodes"] = [[] for _ in G["iset_names"]]
    for n, I in G["iset"].items():
        G["infoset_nodes"][I].append(n)
    avg, hist = cfr_plus(G, iters, log_every=max(1, iters // 20))
    e = exploitability(G, avg)
    v = game_value(G, avg)
    print(f"final exploitability {e:.6e}, game value (P0) {v:.6f}", file=sys.stderr)
    write_strategy(G, avg, out_path,
                   f"CFR+ (simultaneous, linear averaging), {iters} iterations, "
                   f"exploitability {e:.3e}")
    print(json.dumps({"exploitability": e, "value_p0": v, "iterations": iters}))


if __name__ == "__main__":
    main()
