"""CFR+ (regret floored at zero, linear average strategy) for a bluff-game-v1 tree.

Usage: python3 -I solve_leduc.py [game.json] [strategy.json] [iterations] [max_seconds]
Reads the game from game.json, writes the average strategy to strategy.json (bluff-strategy-v1).
Exploitability is computed exactly by best-response traversal of the tree.
"""
import json
import os
import sys
import time
from fractions import Fraction

sys.setrecursionlimit(20000)

HERE = os.path.dirname(os.path.abspath(__file__))
GAME_PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "game.json")
OUT_PATH = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "strategy.json")
ITERS = int(sys.argv[3]) if len(sys.argv) > 3 else 20000
MAX_SECONDS = float(sys.argv[4]) if len(sys.argv) > 4 else 600.0

CHANCE, DECISION, TERMINAL = 0, 1, 2

# ---------------------------------------------------------------- load + validate
with open(GAME_PATH) as f:
    game = json.load(f)
assert game["format"] == "bluff-game-v1", game["format"]
raw_nodes = {n["id"]: n for n in game["nodes"]}
root_id = game["root"]

idx = {nid: i for i, nid in enumerate(raw_nodes)}  # contiguous indices
N = len(raw_nodes)
kind = [None] * N
player = [-1] * N
infoset = [-1] * N
children = [None] * N      # decision: list of child indices in action order
chance_outs = [None] * N   # chance: list of (prob float, child index)
util = [0.0] * N           # terminal: payoff to player 0
parent_count = [0] * N

infoset_names = []
infoset_id = {}
infoset_player = []
infoset_actions = []
infoset_nodes = []

for nid, n in raw_nodes.items():
    i = idx[nid]
    k = n["kind"]
    if k == "terminal":
        kind[i] = TERMINAL
        util[i] = float(Fraction(n["payoff_p0"]))
    elif k == "chance":
        kind[i] = CHANCE
        outs = []
        total = Fraction(0)
        for o in n["outcomes"]:
            p = Fraction(o["prob"])
            total += p
            if p <= 0:
                raise ValueError(f"non-positive chance probability at node {nid}")
            outs.append((p, o["child"]))
        if total != 1:
            raise ValueError(f"chance node {nid} probabilities sum to {total}")
        chance_outs[i] = [(float(p), idx[c]) for p, c in outs]
        for _, c in outs:
            parent_count[idx[c]] += 1
    elif k == "decision":
        kind[i] = DECISION
        player[i] = int(n["player"])
        labels = [a["label"] for a in n["actions"]]
        name = n["infoset"]
        if name not in infoset_id:
            infoset_id[name] = len(infoset_names)
            infoset_names.append(name)
            infoset_player.append(player[i])
            infoset_actions.append(labels)
            infoset_nodes.append([])
        j = infoset_id[name]
        if infoset_player[j] != player[i] or infoset_actions[j] != labels:
            raise ValueError(f"inconsistent infoset {name!r}")
        infoset[i] = j
        infoset_nodes[j].append(i)
        children[i] = [idx[a["child"]] for a in n["actions"]]
        for c in children[i]:
            parent_count[c] += 1
    else:
        raise ValueError(f"unknown node kind {k!r}")

root = idx[root_id]
if parent_count[root] != 0 or any(parent_count[i] != 1 for i in range(N) if i != root):
    raise ValueError("game is not a tree (every non-root node needs exactly one parent)")

NI = len(infoset_names)
NA = [len(a) for a in infoset_actions]
print(f"loaded {game.get('name')}: {N} nodes, {NI} infosets", flush=True)

# ---------------------------------------------------------------- CFR+
regret = [[0.0] * m for m in NA]
strat_sum = [[0.0] * m for m in NA]


def regret_match(j):
    r = regret[j]
    s = sum(r)
    if s > 0:
        return [x / s for x in r]
    return [1.0 / len(r)] * len(r)


def cfr(h, reach0, reach1, rc, t, delta):
    k = kind[h]
    if k == TERMINAL:
        u = util[h]
        return (u, -u)
    if k == CHANCE:
        v0 = v1 = 0.0
        for p, c in chance_outs[h]:
            a0, a1 = cfr(c, reach0, reach1, rc * p, t, delta)
            v0 += p * a0
            v1 += p * a1
        return (v0, v1)
    pl = player[h]
    j = infoset[h]
    sig = regret_match(j)
    own = reach0 if pl == 0 else reach1
    opp = reach1 if pl == 0 else reach0
    w = t * own
    ss = strat_sum[j]
    for a in range(NA[j]):
        ss[a] += w * sig[a]
    ch = children[h]
    cv = []
    for a, c in enumerate(ch):
        if pl == 0:
            cv.append(cfr(c, reach0 * sig[a], reach1, rc, t, delta))
        else:
            cv.append(cfr(c, reach0, reach1 * sig[a], rc, t, delta))
    v0 = sum(sig[a] * cv[a][0] for a in range(len(ch)))
    v1 = sum(sig[a] * cv[a][1] for a in range(len(ch)))
    vp = v0 if pl == 0 else v1
    d = delta[j]
    orc = opp * rc
    for a in range(len(ch)):
        d[a] += orc * (cv[a][pl] - vp)
    return (v0, v1)


def average_strategy():
    avg = []
    for j in range(NI):
        s = sum(strat_sum[j])
        if s > 0:
            avg.append([x / s for x in strat_sum[j]])
        else:
            avg.append([1.0 / NA[j]] * NA[j])
    return avg


def expected_p0(h, avg):
    k = kind[h]
    if k == TERMINAL:
        return util[h]
    if k == CHANCE:
        return sum(p * expected_p0(c, avg) for p, c in chance_outs[h])
    sig = avg[infoset[h]]
    return sum(sig[a] * expected_p0(c, avg) for a, c in enumerate(children[h]))


def best_response_value(b, avg):
    """Expected payoff to player b when b best-responds to the average strategy of the other player."""
    sign = 1.0 if b == 0 else -1.0
    opp = 1 - b
    # forward pass: probability of reaching each node through chance and the opponent only
    weight = [0.0] * N
    weight[root] = 1.0
    order = [root]
    seen = [False] * N
    seen[root] = True
    stack = [root]
    while stack:
        h = stack.pop()
        k = kind[h]
        if k == CHANCE:
            nxt = [(c, weight[h] * p) for p, c in chance_outs[h]]
        elif k == DECISION and player[h] == opp:
            sig = avg[infoset[h]]
            nxt = [(c, weight[h] * sig[a]) for a, c in enumerate(children[h])]
        elif k == DECISION:
            nxt = [(c, weight[h]) for c in children[h]]
        else:
            nxt = []
        for c, w in nxt:
            weight[c] = w
            stack.append(c)
    memo = {}
    best = {}
    in_progress = set()

    def brv(h):
        if h in memo:
            return memo[h]
        k = kind[h]
        if k == TERMINAL:
            res = sign * util[h]
        elif k == CHANCE:
            res = sum(p * brv(c) for p, c in chance_outs[h])
        elif player[h] == opp:
            sig = avg[infoset[h]]
            res = sum(sig[a] * brv(c) for a, c in enumerate(children[h]))
        else:
            j = infoset[h]
            if j not in best:
                if j in in_progress:
                    raise RuntimeError("imperfect recall: cyclic best response")
                in_progress.add(j)
                q = [0.0] * NA[j]
                for hn in infoset_nodes[j]:
                    wt = weight[hn]
                    if wt == 0.0:
                        continue
                    for a, c in enumerate(children[hn]):
                        q[a] += wt * brv(c)
                in_progress.discard(j)
                best[j] = max(range(NA[j]), key=lambda a: (q[a], -a))
            res = brv(children[h][best[j]])
        memo[h] = res
        return res

    return brv(root)


def exploitability(avg):
    e0 = best_response_value(0, avg)
    e1 = best_response_value(1, avg)
    return (e0 + e1) / 2.0, e0, e1


def write_strategy(avg, path, producer):
    probs = {}
    for j, name in enumerate(infoset_names):
        probs[name] = {lab: repr(float(p)) for lab, p in zip(infoset_actions[j], avg[j])}
    out = {
        "format": "bluff-strategy-v1",
        "game": game.get("name"),
        "producer": producer,
        "probs": probs,
    }
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=1)
    os.replace(tmp, path)


# ---------------------------------------------------------------- main loop
start = time.time()
t_done = 0
for t in range(1, ITERS + 1):
    delta = [[0.0] * m for m in NA]
    cfr(root, 1.0, 1.0, 1.0, t, delta)
    for j in range(NI):
        r = regret[j]
        d = delta[j]
        for a in range(NA[j]):
            v = r[a] + d[a]
            r[a] = v if v > 0.0 else 0.0
    t_done = t
    if t % 500 == 0 or t == ITERS or time.time() - start > MAX_SECONDS:
        avg = average_strategy()
        expl, e0, e1 = exploitability(avg)
        print(f"iter {t}: exploitability {expl:.3e} (BR0 {e0:.3e}, BR1 {e1:.3e}) "
              f"elapsed {time.time() - start:.0f}s", flush=True)
        if time.time() - start > MAX_SECONDS:
            print("time limit reached, stopping", flush=True)
            break

avg = average_strategy()
expl, e0, e1 = exploitability(avg)
value_p0 = expected_p0(root, avg)
producer = (f"CFR+ with linear averaging, {t_done} iterations, pure Python, "
            f"exact best-response exploitability {expl:.6g}")
write_strategy(avg, OUT_PATH, producer)
print(f"FINAL iterations={t_done} exploitability={expl:.8g} BR0={e0:.8g} BR1={e1:.8g} "
      f"value_p0={value_p0:.8g}", flush=True)
print(f"wrote {OUT_PATH}", flush=True)
