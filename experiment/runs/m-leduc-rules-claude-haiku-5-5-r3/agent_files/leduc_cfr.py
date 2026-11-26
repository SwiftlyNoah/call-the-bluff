#!/usr/bin/env python3
"""Leduc hold'em: explicit game tree, CFR+ (alternating updates, linear averaging),
exact best-response exploitability, and strategy.json output (bluff-strategy-v1).

Usage: python3 leduc_cfr.py [iterations] [output_path]
"""
import json
import sys
import time

CARDS = ['Js', 'Jh', 'Qs', 'Qh', 'Ks', 'Kh']
RANK = {'J': 0, 'Q': 1, 'K': 2}
LETTER = {'fold': 'f', 'call': 'c', 'raise': 'r'}
RAISE_SIZE = (2, 4)      # round 1, round 2
MAX_RAISES = 2           # per round

# ---------------------------------------------------------------- game tree
TERM, CHANCE, PLAYER = 0, 1, 2
ntype, nchild, nprob, nplayer, niset, nval = [], [], [], [], [], []
iset_index = {}
iset_key, iset_actions, iset_player = [], [], []


def new_node(kind):
    ntype.append(kind)
    nchild.append([])
    nprob.append([])
    nplayer.append(-1)
    niset.append(-1)
    nval.append(0.0)
    return len(ntype) - 1


def get_iset(key, player, actions):
    if key not in iset_index:
        iset_index[key] = len(iset_key)
        iset_key.append(key)
        iset_actions.append(actions)
        iset_player.append(player)
    idx = iset_index[key]
    assert iset_actions[idx] == actions, key   # same name => same legal actions
    assert iset_player[idx] == player, key     # name determines the acting player
    return idx


def showdown_p0(h0, h1, pub, pot_half):
    """Payoff to player 0 at showdown; pot_half is each player's contribution."""
    r0, r1, rp = RANK[h0[0]], RANK[h1[0]], RANK[pub[0]]
    m0, m1 = r0 == rp, r1 == rp
    if m0 and not m1:
        return pot_half
    if m1 and not m0:
        return -pot_half
    if r0 > r1:
        return pot_half
    if r0 < r1:
        return -pot_half
    return 0.0


def build(h, pub, rnd, cur, hist1, c, rr):
    """Decision node for the player to act. cur = this round's actions so far,
    hist1 = round-1 actions (round 2 only), c = [contrib0, contrib1], rr = raises this round."""
    p = len(cur) % 2                     # actions alternate, player 0 opens each round
    facing = c[1 - p] > c[p]
    acts = (['fold'] if facing else []) + ['call']
    if rr < MAX_RAISES:
        acts.append('raise')
    if rnd == 1:
        key = f"{h[p]}:{cur}"
    else:
        key = f"{h[p]}|{pub}:{hist1}/{cur}"
    node = new_node(PLAYER)
    nplayer[node] = p
    niset[node] = get_iset(key, p, acts)
    kids = []
    for a in acts:
        newcur = cur + LETTER[a]
        if a == 'fold':
            child = new_node(TERM)
            nval[child] = -c[0] if p == 0 else c[1]
        elif a == 'call':
            newc = [c[0], c[1]]
            newc[p] = c[1 - p]
            if cur:                        # a call after the round's first action ends the round
                if rnd == 1:
                    child = new_node(CHANCE)
                    rest = [x for x in CARDS if x not in (h[0], h[1])]
                    for card in rest:
                        nprob[child].append(1.0 / len(rest))
                        nchild[child].append(build(h, card, 2, '', newcur, newc, 0))
                else:
                    child = new_node(TERM)
                    nval[child] = showdown_p0(h[0], h[1], pub, newc[0])
            else:                          # check
                child = build(h, pub, rnd, newcur, hist1, newc, rr)
        else:                              # raise
            newc = [c[0], c[1]]
            newc[p] = c[1 - p] + RAISE_SIZE[rnd - 1]
            child = build(h, pub, rnd, newcur, hist1, newc, rr + 1)
        kids.append(child)
    nchild[node] = kids
    return node


def build_game():
    root = new_node(CHANCE)
    for h0 in CARDS:
        for h1 in CARDS:
            if h1 == h0:
                continue
            nprob[root].append(1.0 / 30)
            nchild[root].append(build((h0, h1), None, 1, '', '', [1, 1], 0))
    return root


# ---------------------------------------------------------------- CFR+
def regret_match(regret):
    sig = []
    for R in regret:
        pos = [x if x > 0 else 0.0 for x in R]
        s = sum(pos)
        k = len(R)
        sig.append([x / s for x in pos] if s > 0 else [1.0 / k] * k)
    return sig


def cfr_rec(n, i, pi_i, pi_o, t, sig, regret, strat_sum):
    """Value to player i at node n (conditional on reaching n). pi_i / pi_o are the
    reach probabilities of player i and of everyone else (chance included)."""
    kind = ntype[n]
    if kind == TERM:
        return nval[n] if i == 0 else -nval[n]
    if kind == CHANCE:
        v = 0.0
        for ch, pr in zip(nchild[n], nprob[n]):
            v += pr * cfr_rec(ch, i, pi_i, pi_o * pr, t, sig, regret, strat_sum)
        return v
    I = niset[n]
    s = sig[I]
    kids = nchild[n]
    if nplayer[n] == i:
        vals = [cfr_rec(kids[a], i, pi_i * s[a], pi_o, t, sig, regret, strat_sum)
                for a in range(len(kids))]
        v = sum(s[a] * vals[a] for a in range(len(kids)))
        R = regret[I]
        for a in range(len(kids)):
            R[a] = max(R[a] + pi_o * (vals[a] - v), 0.0)      # CFR+: floor at zero
        S = strat_sum[I]
        for a in range(len(kids)):
            S[a] += t * pi_i * s[a]                          # linear averaging
        return v
    v = 0.0
    for a in range(len(kids)):
        v += s[a] * cfr_rec(kids[a], i, pi_i, pi_o * s[a], t, sig, regret, strat_sum)
    return v


def average_strategy(strat_sum):
    out = []
    for S in strat_sum:
        tot = sum(S)
        k = len(S)
        out.append([x / tot for x in S] if tot > 0 else [1.0 / k] * k)
    return out


def fixed_value(n, strat):
    """Expected payoff to player 0 when both players follow strat."""
    kind = ntype[n]
    if kind == TERM:
        return nval[n]
    if kind == CHANCE:
        return sum(pr * fixed_value(ch, strat) for ch, pr in zip(nchild[n], nprob[n]))
    s = strat[niset[n]]
    return sum(s[a] * fixed_value(ch, strat) for a, ch in enumerate(nchild[n]))


def best_response_value(root, i, strat):
    """Exact value to player i of a best response to the opponent's strategy `strat`."""
    N = len(ntype)
    pio = [0.0] * N
    depth = [0] * N
    pio[root] = 1.0
    stack = [root]
    own_nodes = {}                       # iset -> list of nodes
    order = []
    while stack:
        n = stack.pop()
        order.append(n)
        kind = ntype[n]
        if kind == TERM:
            continue
        if kind == CHANCE:
            for ch, pr in zip(nchild[n], nprob[n]):
                pio[ch] = pio[n] * pr
                depth[ch] = depth[n] + 1
                stack.append(ch)
            continue
        I = niset[n]
        s = strat[I] if nplayer[n] != i else None
        if nplayer[n] == i:
            own_nodes.setdefault(I, []).append(n)
        for a, ch in enumerate(nchild[n]):
            pio[ch] = pio[n] if s is None else pio[n] * s[a]
            depth[ch] = depth[n] + 1
            stack.append(ch)

    # decide best responses bottom-up: deepest information sets first
    choice = {}
    W = [None] * N

    def value(n):
        if W[n] is not None:
            return W[n]
        kind = ntype[n]
        if kind == TERM:
            v = nval[n] if i == 0 else -nval[n]
        elif kind == CHANCE:
            v = sum(pr * value(ch) for ch, pr in zip(nchild[n], nprob[n]))
        elif nplayer[n] == i:
            v = value(nchild[n][choice[niset[n]]])
        else:
            s = strat[niset[n]]
            v = sum(s[a] * value(ch) for a, ch in enumerate(nchild[n]))
        W[n] = v
        return v

    isets = sorted(own_nodes, key=lambda I: -depth[own_nodes[I][0]])
    for I in isets:
        nodes = own_nodes[I]
        k = len(nchild[nodes[0]])
        S = [0.0] * k
        for h in nodes:
            for a in range(k):
                S[a] += pio[h] * value(nchild[h][a])
        choice[I] = max(range(k), key=lambda a: S[a])
    return value(root)


def main():
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    out_path = sys.argv[2] if len(sys.argv) > 2 else 'strategy.json'

    root = build_game()
    NI = len(iset_key)
    print(f"tree nodes: {len(ntype)}, information sets: {NI}")

    regret = [[0.0] * len(a) for a in iset_actions]
    strat_sum = [[0.0] * len(a) for a in iset_actions]

    t0 = time.time()
    for t in range(1, iters + 1):
        for i in (0, 1):                 # alternating updates
            sig = regret_match(regret)
            cfr_rec(root, i, 1.0, 1.0, t, sig, regret, strat_sum)
        if t % max(1, iters // 10) == 0:
            avg = average_strategy(strat_sum)
            e = (best_response_value(root, 0, avg) + best_response_value(root, 1, avg)) / 2
            print(f"iter {t:6d}  exploitability {e:.3e}  ({time.time() - t0:.1f}s)", flush=True)

    avg = average_strategy(strat_sum)
    br0 = best_response_value(root, 0, avg)
    br1 = best_response_value(root, 1, avg)
    expl = (br0 + br1) / 2
    value_p0 = fixed_value(root, avg)
    print(f"final exploitability {expl:.6e}  (BR0 {br0:.6e}, BR1 {br1:.6e})")
    print(f"value to player 0: {value_p0:.8f}")

    probs = {}
    for I, key in enumerate(iset_key):
        probs[key] = {a: repr(float(p)) for a, p in zip(iset_actions[I], avg[I])}
        assert abs(sum(avg[I]) - 1.0) < 1e-9
    doc = {
        "format": "bluff-strategy-v1",
        "game": "leduc",
        "producer": f"CFR+ (alternating updates, linear averaging), {iters} iterations, exact full-tree traversal; exploitability computed by exact best response",
        "probs": probs,
    }
    with open(out_path, 'w') as f:
        json.dump(doc, f, indent=1)
    print(f"wrote {out_path} with {len(probs)} information sets")
    print(json.dumps({"claim_converged": expl < 1e-3, "claimed_exploitability": expl,
                      "claimed_value_p0": value_p0, "exploitability_check": "exact"}))


if __name__ == '__main__':
    main()
