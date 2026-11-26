#!/usr/bin/env python3
"""Solve Leduc hold'em with CFR+ (alternating updates, linear averaging).

Builds the full game tree (all 120 ordered deals of Js..Kh), runs CFR+,
computes exploitability with exact best responses, and writes strategy.json
in the bluff-strategy-v1 format.

Usage: python3 -I leduc_cfr.py [iterations] [output_path]
"""
import itertools
import json
import sys
import time

CARDS = ["Js", "Jh", "Qs", "Qh", "Ks", "Kh"]
RANK = {"J": 0, "Q": 1, "K": 2}
RAISE_SIZE = {1: 2, 2: 4}  # round number -> raise size
MAX_RAISES = 2
LABEL = {"f": "fold", "c": "call", "r": "raise"}
DEALS = list(itertools.permutations(CARDS, 3))  # (player0 card, player1 card, public card)
DEAL_PROB = 1.0 / len(DEALS)


def contribs(acts, size):
    """Chips each player has put into the current betting round (index = player)."""
    c = [0, 0]
    for i, a in enumerate(acts):
        p = i % 2
        if a == "c":
            c[p] = max(c)
        elif a == "r":
            c[p] = max(c) + size
    return c


def round_status(acts):
    """'open', 'end' (check/call after the round's first action) or 'fold'."""
    if not acts:
        return "open"
    if acts[-1] == "f":
        return "fold"
    if acts[-1] == "c" and len(acts) >= 2:
        return "end"
    return "open"


def terminal_u0(deal, r1, r2):
    """Payoff to player 0 at a terminal node (r2 is None when round 1 folded)."""
    c0, c1, pub = deal
    a1 = contribs(r1, RAISE_SIZE[1])
    tot = [1 + a1[0], 1 + a1[1]]  # antes included
    if r2 is None:
        folder = (len(r1) - 1) % 2
    else:
        a2 = contribs(r2, RAISE_SIZE[2])
        tot = [tot[0] + a2[0], tot[1] + a2[1]]
        folder = (len(r2) - 1) % 2 if round_status(r2) == "fold" else None
    if folder == 0:
        return -tot[0]
    if folder == 1:
        return tot[1]
    # Showdown: a private card matching the public rank wins; otherwise higher rank.
    if c0[0] == pub[0]:
        return tot[1]
    if c1[0] == pub[0]:
        return -tot[0]
    r0, rr1 = RANK[c0[0]], RANK[c1[0]]
    if r0 > rr1:
        return tot[1]
    if r0 < rr1:
        return -tot[0]
    return 0.0


# ---------------------------------------------------------------- game tree
nodes = []  # dict per node; terminal nodes carry "u0", others carry p/info/children/letters


def build(deal, r1, r2):
    if r2 is None and round_status(r1) == "end":
        r2 = ""  # round 1 ended by a call/check: round 2 starts (no action yet)
    idx = len(nodes)
    nodes.append(None)
    rnd = 1 if r2 is None else 2
    acts = r1 if rnd == 1 else r2
    status = round_status(acts)
    depth = len(r1) + (len(r2) if r2 else 0)
    if status == "fold" or (rnd == 2 and status == "end"):
        nodes[idx] = {"terminal": True, "u0": terminal_u0(deal, r1, r2), "depth": depth}
        return idx
    p = len(acts) % 2
    c = contribs(acts, RAISE_SIZE[rnd])
    owed = max(c) - c[p]
    letters = []
    if owed > 0:
        letters.append("f")
    letters.append("c")  # call, or check when nothing is owed
    if acts.count("r") < MAX_RAISES:
        letters.append("r")
    card = deal[p]
    if rnd == 1:
        key = f"{card}:{r1}"
    else:
        key = f"{card}|{deal[2]}:{r1}/{r2}"
    children = []
    for a in letters:
        if rnd == 1:
            children.append(build(deal, r1 + a, None))
        else:
            children.append(build(deal, r1, r2 + a))
    nodes[idx] = {"terminal": False, "p": p, "key": key, "letters": letters,
                  "children": children, "depth": depth}
    return idx


roots = [build(d, "", None) for d in DEALS]

# Information sets: one id per key; a key always belongs to one player and one depth.
INFO_KEYS = []
INFO_LETTERS = []
info_id = {}
for nd in nodes:
    if nd["terminal"] or nd["key"] in info_id:
        continue
    info_id[nd["key"]] = len(INFO_KEYS)
    INFO_KEYS.append(nd["key"])
    INFO_LETTERS.append(nd["letters"])
for nd in nodes:
    if not nd["terminal"]:
        nd["info"] = info_id[nd["key"]]
        nd["letters"] = tuple(nd["letters"])

MAX_DEPTH = max(nd["depth"] for nd in nodes)
BY_DEPTH = [[] for _ in range(MAX_DEPTH + 1)]
for i, nd in enumerate(nodes):
    BY_DEPTH[nd["depth"]].append(i)

# Regret and strategy-sum tables, one list per information set.
REG = [[0.0] * len(l) for l in INFO_LETTERS]
SUM = [[0.0] * len(l) for l in INFO_LETTERS]


def regret_matching(reg):
    pos = [r if r > 0.0 else 0.0 for r in reg]
    s = sum(pos)
    if s > 0.0:
        return [x / s for x in pos]
    return [1.0 / len(reg)] * len(reg)


def cfr(n, pi0, pi1, c, u, t):
    """CFR+ traversal. Returns the value to player 0 of node n.

    pi0, pi1: reach probabilities of the two players; c: chance (and deal) weight.
    Only player u's regrets and strategy sums are updated in this traversal.
    """
    nd = nodes[n]
    if nd["terminal"]:
        return nd["u0"]
    p = nd["p"]
    children = nd["children"]
    iid = nd["info"]
    sigma = regret_matching(REG[iid])
    vals = []
    for k, child in enumerate(children):
        if p == 0:
            vals.append(cfr(child, pi0 * sigma[k], pi1, c, u, t))
        else:
            vals.append(cfr(child, pi0, pi1 * sigma[k], c, u, t))
    v = 0.0
    for k in range(len(children)):
        v += sigma[k] * vals[k]
    if p == u:
        sign = 1.0 if p == 0 else -1.0
        opp = (pi1 if p == 0 else pi0) * c
        own = pi0 if p == 0 else pi1
        util = sign * v
        reg = REG[iid]
        ssum = SUM[iid]
        for k in range(len(children)):
            reg[k] = max(reg[k] + opp * (sign * vals[k] - util), 0.0)
            ssum[k] += t * own * sigma[k]
    return v


def average_profile():
    prof = []
    for iid in range(len(INFO_KEYS)):
        s = SUM[iid]
        tot = sum(s)
        if tot > 0.0:
            prof.append([x / tot for x in s])
        else:
            prof.append([1.0 / len(s)] * len(s))
    return prof


def game_value(prof, br=None):
    """Expected payoff under prof.

    br None: value to player 0 when both players play prof.
    br 0 or 1: best-response value to player br against the other player's prof.
    """
    sgn = -1.0 if br == 1 else 1.0
    V = [0.0] * len(nodes)
    reach = None
    if br is not None:
        reach = [0.0] * len(nodes)
        for r in roots:
            reach[r] = DEAL_PROB
        for d in range(MAX_DEPTH + 1):
            for i in BY_DEPTH[d]:
                nd = nodes[i]
                if nd["terminal"]:
                    continue
                pr = reach[i]
                opp_move = nd["p"] != br
                for k, ch in enumerate(nd["children"]):
                    reach[ch] += pr * prof[nd["info"]][k] if opp_move else pr
    choice = {}
    for d in range(MAX_DEPTH, -1, -1):
        if br is not None:
            Q = {}
            for i in BY_DEPTH[d]:
                nd = nodes[i]
                if nd["terminal"] or nd["p"] != br:
                    continue
                q = Q.setdefault(nd["info"], [0.0] * len(nd["children"]))
                for k, ch in enumerate(nd["children"]):
                    q[k] += reach[i] * V[ch]
            for iid, q in Q.items():
                choice[iid] = max(range(len(q)), key=lambda k: q[k])
        for i in BY_DEPTH[d]:
            nd = nodes[i]
            if nd["terminal"]:
                V[i] = sgn * nd["u0"]
            elif br is not None and nd["p"] == br:
                V[i] = V[nd["children"][choice[nd["info"]]]]
            else:
                sg = prof[nd["info"]]
                V[i] = sum(sg[k] * V[ch] for k, ch in enumerate(nd["children"]))
    return sum(DEAL_PROB * V[r] for r in roots)


def exploitability(prof):
    br0 = game_value(prof, br=0)  # player 0's best-response payoff
    br1 = game_value(prof, br=1)  # player 1's best-response payoff
    return (br0 + br1) / 2.0, br0, br1


def write_strategy(prof, path):
    probs = {}
    for iid, key in enumerate(INFO_KEYS):
        letters = INFO_LETTERS[iid]
        probs[key] = {LABEL[a]: repr(float(prof[iid][k])) for k, a in enumerate(letters)}
    out = {
        "format": "bluff-strategy-v1",
        "game": "leduc",
        "producer": "CFR+ (alternating updates, linear averaging), full-tree enumeration "
                    "of all 120 deals, exact best-response exploitability",
        "probs": probs,
    }
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
        f.write("\n")


def main():
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    out_path = sys.argv[2] if len(sys.argv) > 2 else "strategy.json"
    print(f"nodes={len(nodes)} info_sets={len(INFO_KEYS)} deals={len(DEALS)}", flush=True)
    start = time.time()
    for t in range(1, iters + 1):
        for u in (0, 1):
            for root in roots:
                cfr(root, 1.0, 1.0, DEAL_PROB, u, t)
        if t in (10, 100) or t % 500 == 0 or t == iters:
            prof = average_profile()
            expl, br0, br1 = exploitability(prof)
            print(f"iter {t:6d}  exploitability={expl:.8f}  "
                  f"(br0={br0:.8f} br1={br1:.8f})  elapsed={time.time() - start:.1f}s",
                  flush=True)
    prof = average_profile()
    expl, br0, br1 = exploitability(prof)
    value_p0 = game_value(prof)
    write_strategy(prof, out_path)
    print(f"FINAL exploitability={expl:.10f} br0={br0:.10f} br1={br1:.10f} "
          f"value_p0={value_p0:.10f}", flush=True)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
