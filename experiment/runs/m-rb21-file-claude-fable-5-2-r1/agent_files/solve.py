#!/usr/bin/env python3
"""CFR+ solver for games in the bluff-game-v1 format.

Reads game.json, runs CFR+ (alternating updates, regret matching+, linearly
weighted average strategy), writes the average strategy to strategy.json, then
re-reads that file and reports its exact exploitability and value (computed
with rational arithmetic on the strategy exactly as written).

Usage: python3 solve.py [--game game.json] [--out strategy.json]
                        [--iters N] [--target EPS] [--time SECONDS]
"""
import argparse
import json
import sys
import time
from fractions import Fraction

SCALE = 10 ** 12  # strategy probabilities are written as exact multiples of 1/SCALE


class Game:
    def __init__(self, path):
        with open(path) as f:
            raw = json.load(f)
        self.name = raw.get("name", "game")
        nodes = raw["nodes"]
        ids = [n["id"] for n in nodes]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate node ids")
        index = {nid: i for i, nid in enumerate(ids)}
        n = len(nodes)
        self.n = n
        self.root = index[raw["root"]]
        self.kind = [0] * n            # 0 terminal, 1 chance, 2 decision
        self.player = [-1] * n
        self.children = [()] * n
        self.labels = [()] * n
        self.payoff_exact = [None] * n
        self.payoff = [0.0] * n
        self.chance_exact = [()] * n
        self.chance = [()] * n
        self.infoset_of = [-1] * n
        self.infoset_names = []
        self.infoset_player = []
        self.infoset_actions = []
        self.infoset_nodes = []
        name_to_infoset = {}
        for i, nd in enumerate(nodes):
            kind = nd["kind"]
            if kind == "terminal":
                self.payoff_exact[i] = Fraction(nd["payoff_p0"])
                self.payoff[i] = float(self.payoff_exact[i])
            elif kind == "chance":
                self.kind[i] = 1
                self.children[i] = tuple(index[o["child"]] for o in nd["outcomes"])
                self.labels[i] = tuple(o["label"] for o in nd["outcomes"])
                probs = tuple(Fraction(o["prob"]) for o in nd["outcomes"])
                if sum(probs) != 1 or min(probs) < 0:
                    raise ValueError("chance node %r: bad probabilities" % nd["id"])
                self.chance_exact[i] = probs
                self.chance[i] = tuple(float(p) for p in probs)
            elif kind == "decision":
                self.kind[i] = 2
                player = nd["player"]
                if player not in (0, 1):
                    raise ValueError("node %r: bad player" % nd["id"])
                self.player[i] = player
                self.children[i] = tuple(index[a["child"]] for a in nd["actions"])
                labels = tuple(a["label"] for a in nd["actions"])
                if len(set(labels)) != len(labels) or not labels:
                    raise ValueError("node %r: bad action labels" % nd["id"])
                self.labels[i] = labels
                name = nd["infoset"]
                k = name_to_infoset.get(name)
                if k is None:
                    k = len(self.infoset_names)
                    name_to_infoset[name] = k
                    self.infoset_names.append(name)
                    self.infoset_player.append(player)
                    self.infoset_actions.append(labels)
                    self.infoset_nodes.append([])
                elif self.infoset_player[k] != player or self.infoset_actions[k] != labels:
                    raise ValueError("infoset %r is inconsistent across its nodes" % name)
                self.infoset_of[i] = k
                self.infoset_nodes[k].append(i)
            else:
                raise ValueError("node %r: unknown kind %r" % (nd["id"], kind))
        # Topological (parents first) order, also checks the graph is a tree.
        order = []
        seen = [False] * n
        stack = [self.root]
        while stack:
            u = stack.pop()
            if seen[u]:
                raise ValueError("node reached twice: not a tree")
            seen[u] = True
            order.append(u)
            stack.extend(reversed(self.children[u]))
        if len(order) != n:
            raise ValueError("%d nodes are unreachable from the root" % (n - len(order)))
        self.order = order
        self.num_infosets = len(self.infoset_names)

    def perfect_recall_violations(self):
        """Infosets whose nodes do not share the same own (infoset, action) history."""
        hist = [None] * self.n
        hist[self.root] = ((), ())
        for u in self.order:
            h = hist[u]
            if self.kind[u] == 2:
                p = self.player[u]
                k = self.infoset_of[u]
                for a, c in enumerate(self.children[u]):
                    hc = list(h)
                    hc[p] = h[p] + ((k, a),)
                    hist[c] = tuple(hc)
            else:
                for c in self.children[u]:
                    hist[c] = h
        bad = []
        for k, members in enumerate(self.infoset_nodes):
            p = self.infoset_player[k]
            if len({hist[u][p] for u in members}) > 1:
                bad.append(self.infoset_names[k])
        return bad


def cfr_plus(game, max_iters, target, time_limit, check_every=250, log=None):
    """CFR+ with alternating updates. Returns (average strategy, iterations run)."""
    kind, player, children = game.kind, game.player, game.children
    infoset_of, chance, payoff = game.infoset_of, game.chance, game.payoff
    order = game.order
    rev_order = order[::-1]
    num_i = game.num_infosets
    sizes = [len(a) for a in game.infoset_actions]
    regret = [[0.0] * s for s in sizes]
    strat_sum = [[0.0] * s for s in sizes]
    sigma = [[1.0 / s] * s for s in sizes]
    first_node = [m[0] for m in game.infoset_nodes]
    infosets_of_player = [[k for k in range(num_i) if game.infoset_player[k] == p] for p in (0, 1)]
    n = game.n
    r_own = [0.0] * n   # reach probability contributed by the updating player
    r_opp = [0.0] * n   # reach probability contributed by the opponent and chance
    val = [0.0] * n     # expected payoff to the updating player
    start = time.time()
    t = 0
    while t < max_iters:
        t += 1
        for p in (0, 1):
            sign = 1.0 if p == 0 else -1.0
            r_own[game.root] = 1.0
            r_opp[game.root] = 1.0
            for u in order:
                k = kind[u]
                if k == 0:
                    continue
                ro, rc = r_own[u], r_opp[u]
                ch = children[u]
                if k == 1:
                    pr = chance[u]
                    for a in range(len(ch)):
                        c = ch[a]
                        r_own[c] = ro
                        r_opp[c] = rc * pr[a]
                else:
                    s = sigma[infoset_of[u]]
                    if player[u] == p:
                        for a in range(len(ch)):
                            c = ch[a]
                            r_own[c] = ro * s[a]
                            r_opp[c] = rc
                    else:
                        for a in range(len(ch)):
                            c = ch[a]
                            r_own[c] = ro
                            r_opp[c] = rc * s[a]
            for u in rev_order:
                k = kind[u]
                if k == 0:
                    val[u] = sign * payoff[u]
                    continue
                ch = children[u]
                if k == 1:
                    pr = chance[u]
                    v = 0.0
                    for a in range(len(ch)):
                        v += pr[a] * val[ch[a]]
                    val[u] = v
                else:
                    i = infoset_of[u]
                    s = sigma[i]
                    v = 0.0
                    for a in range(len(ch)):
                        v += s[a] * val[ch[a]]
                    val[u] = v
                    if player[u] == p:
                        rc = r_opp[u]
                        if rc > 0.0:
                            r = regret[i]
                            for a in range(len(ch)):
                                r[a] += rc * (val[ch[a]] - v)
            # Average strategy (linear weights), then regret matching+.
            for i in infosets_of_player[p]:
                w = t * r_own[first_node[i]]
                s = sigma[i]
                if w > 0.0:
                    acc = strat_sum[i]
                    for a in range(len(s)):
                        acc[a] += w * s[a]
                r = regret[i]
                tot = 0.0
                for a in range(len(r)):
                    if r[a] < 0.0:
                        r[a] = 0.0
                    else:
                        tot += r[a]
                if tot > 0.0:
                    for a in range(len(r)):
                        s[a] = r[a] / tot
                else:
                    u = 1.0 / len(r)
                    for a in range(len(r)):
                        s[a] = u
        if t % check_every == 0 or t == max_iters:
            avg = average_strategy(strat_sum)
            expl = exploitability(game, avg, exact=False)[0]
            if log:
                log("iter %6d  exploitability %.3e  (%.1fs)" % (t, expl, time.time() - start))
            if expl <= target or time.time() - start > time_limit:
                break
    return average_strategy(strat_sum), t


def average_strategy(strat_sum):
    out = []
    for acc in strat_sum:
        tot = sum(acc)
        if tot > 0.0:
            out.append([x / tot for x in acc])
        else:  # never reached by its own player: any choice is fine
            out.append([1.0 / len(acc)] * len(acc))
    return out


def best_response_value(game, strat, br_player, exact):
    """Value to br_player of a best response against strat (the opponent's part is used).

    Works bottom-up over the tree; at each of br_player's infosets the action that
    maximizes the counterfactual value summed over the infoset's nodes is chosen.
    Requires perfect recall for br_player.
    """
    zero = Fraction(0) if exact else 0.0
    one = Fraction(1) if exact else 1.0
    chance = game.chance_exact if exact else game.chance
    payoff = game.payoff_exact if exact else game.payoff
    kind, player, children, infoset_of = game.kind, game.player, game.children, game.infoset_of
    n = game.n
    reach = [zero] * n  # opponent and chance reach
    reach[game.root] = one
    depth = [0] * n
    for u in game.order:
        k = kind[u]
        if k == 0:
            continue
        ch = children[u]
        for a, c in enumerate(ch):
            depth[c] = depth[u] + 1
            if k == 1:
                reach[c] = reach[u] * chance[u][a]
            elif player[u] == br_player:
                reach[c] = reach[u]
            else:
                reach[c] = reach[u] * strat[infoset_of[u]][a]
    val = [None] * n
    choice = {}
    sign = 1 if br_player == 0 else -1

    def resolve(i):
        members = game.infoset_nodes[i]
        best, best_a = None, 0
        for a in range(len(game.infoset_actions[i])):
            tot = zero
            for u in members:
                tot += reach[u] * value(children[u][a])
            if best is None or tot > best:
                best, best_a = tot, a
        choice[i] = best_a
        return best_a

    def value(u):
        # Iterative post-order evaluation with memoization.
        if val[u] is not None:
            return val[u]
        stack = [u]
        while stack:
            x = stack[-1]
            if val[x] is not None:
                stack.pop()
                continue
            k = kind[x]
            if k == 0:
                val[x] = sign * payoff[x]
                stack.pop()
                continue
            ch = children[x]
            if k == 2 and player[x] == br_player:
                i = infoset_of[x]
                if i not in choice:
                    if i in resolving:
                        raise ValueError("imperfect recall: cannot compute a best response")
                    resolving.add(i)
                    resolve(i)
                    resolving.discard(i)
                c = ch[choice[i]]
                if val[c] is None:
                    stack.append(c)
                    continue
                val[x] = val[c]
                stack.pop()
                continue
            pending = [c for c in ch if val[c] is None]
            if pending:
                stack.extend(pending)
                continue
            pr = chance[x] if k == 1 else strat[infoset_of[x]]
            v = zero
            for a, c in enumerate(ch):
                v += pr[a] * val[c]
            val[x] = v
            stack.pop()
        return val[u]

    resolving = set()
    return value(game.root)


def expected_value(game, strat, exact):
    """Expected payoff to player 0 when both players follow strat."""
    chance = game.chance_exact if exact else game.chance
    payoff = game.payoff_exact if exact else game.payoff
    val = [None] * game.n
    for u in reversed(game.order):
        k = game.kind[u]
        if k == 0:
            val[u] = payoff[u]
        else:
            pr = chance[u] if k == 1 else strat[game.infoset_of[u]]
            v = 0
            for a, c in enumerate(game.children[u]):
                v += pr[a] * val[c]
            val[u] = v
    return val[game.root]


def exploitability(game, strat, exact):
    br0 = best_response_value(game, strat, 0, exact)
    br1 = best_response_value(game, strat, 1, exact)
    return (br0 + br1) / 2, br0, br1


def quantize(probs):
    """Round to exact multiples of 1/SCALE that sum to exactly 1."""
    ints = [max(0, int(round(p * SCALE))) for p in probs]
    j = max(range(len(ints)), key=lambda a: ints[a])
    ints[j] += SCALE - sum(ints)
    return ints


def format_prob(x):
    if x == SCALE:
        return "1"
    s = ("%013d" % x)
    s = s[:-12] + "." + s[-12:]
    s = s.rstrip("0")
    return s + "0" if s.endswith(".") else s


def write_strategy(game, strat, path, producer):
    probs = {}
    for i, name in enumerate(game.infoset_names):
        ints = quantize(strat[i])
        probs[name] = {lab: format_prob(x) for lab, x in zip(game.infoset_actions[i], ints)}
    doc = {"format": "bluff-strategy-v1", "game": game.name, "producer": producer, "probs": probs}
    with open(path, "w") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")


def read_strategy(game, path):
    """Read a strategy file back as exact fractions, validating it against the game."""
    with open(path) as f:
        doc = json.load(f)
    probs = doc["probs"]
    if set(probs) != set(game.infoset_names):
        raise ValueError("strategy file infosets do not match the game")
    strat = []
    for i, name in enumerate(game.infoset_names):
        entry = probs[name]
        if set(entry) != set(game.infoset_actions[i]):
            raise ValueError("infoset %r: actions do not match the game" % name)
        row = [Fraction(entry[lab]) for lab in game.infoset_actions[i]]
        if sum(row) != 1 or min(row) < 0:
            raise ValueError("infoset %r: probabilities are not a distribution" % name)
        strat.append(row)
    return strat


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--game", default="game.json")
    ap.add_argument("--out", default="strategy.json")
    ap.add_argument("--iters", type=int, default=200000, help="maximum CFR+ iterations")
    ap.add_argument("--target", type=float, default=1e-6, help="stop once exploitability is at most this")
    ap.add_argument("--time", type=float, default=600.0, help="stop after about this many seconds")
    args = ap.parse_args()

    def log(msg):
        print(msg, file=sys.stderr, flush=True)

    game = Game(args.game)
    log("game %r: %d nodes, %d infosets" % (game.name, game.n, game.num_infosets))
    bad = game.perfect_recall_violations()
    if bad:
        log("WARNING: %d infosets violate perfect recall, e.g. %r" % (len(bad), bad[0]))
    else:
        log("perfect recall: ok")

    strat, iters = cfr_plus(game, args.iters, args.target, args.time, log=log)
    producer = "solve.py: CFR+ (alternating updates, linear averaging), %d iterations" % iters
    write_strategy(game, strat, args.out, producer)

    written = read_strategy(game, args.out)
    expl, br0, br1 = exploitability(game, written, exact=True)
    value = expected_value(game, written, exact=True)
    result = {
        "iterations": iters,
        "perfect_recall": not bad,
        "exploitability": float(expl),
        "br_value_p0": float(br0),
        "br_value_p1": float(br1),
        "value_p0": float(value),
        "value_p0_bounds": [float(-br1), float(br0)],
    }
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
