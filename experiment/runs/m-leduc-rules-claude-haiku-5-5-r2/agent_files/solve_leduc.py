"""Leduc hold'em solver: CFR+ (alternating updates, linear averaging) on the full game tree.

Writes strategy.json in the bluff-strategy-v1 format and reports exploitability
(computed exactly with best responses over the full tree).
"""
import json
import sys
import time

CARDS = ['Js', 'Jh', 'Qs', 'Qh', 'Ks', 'Kh']
RANK = {'J': 0, 'Q': 1, 'K': 2}
RAISE = {1: 2, 2: 4}
MAX_RAISES = 2
LABEL = {'f': 'fold', 'c': 'call', 'r': 'raise'}

ITERATIONS = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
OUT_PATH = sys.argv[2] if len(sys.argv) > 2 else 'strategy.json'

# ---------------------------------------------------------------- game tree
nodes = []      # node dicts; ids are allocated before children, so parent id < child id
infosets = {}   # name -> index
IS = []         # per information set: name, player, letters, depth, nodes


def new_node():
    nodes.append(None)
    return len(nodes) - 1


def infoset(name, player, letters, depth):
    if name not in infosets:
        infosets[name] = len(IS)
        IS.append({'name': name, 'player': player, 'letters': letters,
                   'depth': depth, 'nodes': []})
    idx = infosets[name]
    info = IS[idx]
    assert info['player'] == player and info['letters'] == letters and info['depth'] == depth, name
    return idx


def showdown_payoff_p0(cards, pub, contrib):
    r0, r1, rp = RANK[cards[0][0]], RANK[cards[1][0]], RANK[pub[0]]
    w0, w1 = r0 == rp, r1 == rp
    if w0 and not w1:
        winner = 0
    elif w1 and not w0:
        winner = 1
    elif r0 > r1:
        winner = 0
    elif r1 > r0:
        winner = 1
    else:
        winner = None
    if winner == 0:
        return float(contrib[1])
    if winner == 1:
        return -float(contrib[0])
    return 0.0


def terminal(u):
    nid = new_node()
    nodes[nid] = {'kind': 'T', 'u': u}
    return nid


def betting(cards, pub, rnd, hist1, hist2, contrib, depth):
    """A betting state. rnd is 1 or 2; hist1 is round-1 actions, hist2 round-2 actions."""
    hist = hist1 if rnd == 1 else hist2
    nid = new_node()
    player = len(hist) % 2  # player 0 acts first in each round, and players alternate
    owed = max(contrib) - contrib[player]
    raises = hist.count('r')
    letters = (['f'] if owed > 0 else []) + ['c'] + (['r'] if raises < MAX_RAISES else [])
    if rnd == 1:
        name = f"{cards[player]}:{hist1}"
    else:
        name = f"{cards[player]}|{pub}:{hist1}/{hist2}"
    iid = infoset(name, player, ''.join(letters), depth)

    children = []
    for L in letters:
        if L == 'f':
            # Folding player forfeits their chips to the opponent.
            u = -float(contrib[0]) if player == 0 else float(contrib[1])
            children.append(terminal(u))
        elif L == 'c':
            newc = list(contrib)
            newc[player] = max(contrib)
            round_over = len(hist) >= 1  # check/call after the round's first action
            if not round_over:
                if rnd == 1:
                    children.append(betting(cards, pub, 1, hist1 + 'c', hist2, newc, depth + 1))
                else:
                    children.append(betting(cards, pub, 2, hist1, hist2 + 'c', newc, depth + 1))
            elif rnd == 1:
                # Public card: a chance node, allocated before its children (parents precede children).
                remaining = [c for c in CARDS if c not in cards]
                wrap = new_node()
                pub_children = [
                    (1.0 / len(remaining),
                     betting(cards, p, 2, hist1 + 'c', '', newc, depth + 2))
                    for p in remaining
                ]
                nodes[wrap] = {'kind': 'C', 'children': pub_children}
                children.append(wrap)
            else:
                children.append(terminal(showdown_payoff_p0(cards, pub, newc)))
        else:  # raise
            newc = list(contrib)
            newc[player] = max(contrib) + RAISE[rnd]
            if rnd == 1:
                child = betting(cards, pub, 1, hist1 + 'r', hist2, newc, depth + 1)
            else:
                child = betting(cards, pub, 2, hist1, hist2 + 'r', newc, depth + 1)
            children.append(child)

    nodes[nid] = {'kind': 'P', 'player': player, 'iset': iid, 'letters': letters,
                  'children': children}
    IS[iid]['nodes'].append(nid)
    return nid


def build():
    # The root is allocated first so that it has id 0 (parents precede children).
    root = new_node()
    deals = [(a, b) for a in CARDS for b in CARDS if a != b]
    deal_children = [(1.0 / len(deals), betting((a, b), None, 1, '', '', [1, 1], 1))
                     for a, b in deals]
    nodes[root] = {'kind': 'C', 'children': deal_children}
    return root


# ---------------------------------------------------------------- CFR+
def strategy_from(regret):
    total = sum(regret)
    if total > 0:
        return [x / total for x in regret]
    return [1.0 / len(regret)] * len(regret)


def walk(nid, p0, p1, c, upd, t, regret, strat_sum):
    n = nodes[nid]
    kind = n['kind']
    if kind == 'T':
        return n['u']
    if kind == 'C':
        return sum(p * walk(ch, p0, p1, c * p, upd, t, regret, strat_sum)
                   for p, ch in n['children'])
    pl = n['player']
    iid = n['iset']
    s = strategy_from(regret[iid])
    children = n['children']
    if pl == upd:
        vals = []
        for a, ch in enumerate(children):
            if pl == 0:
                vals.append(walk(ch, p0 * s[a], p1, c, upd, t, regret, strat_sum))
            else:
                vals.append(walk(ch, p0, p1 * s[a], c, upd, t, regret, strat_sum))
        v = sum(s[a] * vals[a] for a in range(len(children)))
        sign = 1.0 if pl == 0 else -1.0  # utility of the updating player
        opp = p1 if pl == 0 else p0
        own = p0 if pl == 0 else p1
        r = regret[iid]
        st = strat_sum[iid]
        for a in range(len(children)):
            r[a] = max(r[a] + c * opp * sign * (vals[a] - v), 0.0)
            st[a] += t * c * own * s[a]
        return v
    v = 0.0
    for a, ch in enumerate(children):
        if pl == 0:
            v += s[a] * walk(ch, p0 * s[a], p1, c, upd, t, regret, strat_sum)
        else:
            v += s[a] * walk(ch, p0, p1 * s[a], c, upd, t, regret, strat_sum)
    return v


# ---------------------------------------------------------------- evaluation
def average_strategy(strat_sum):
    return {i: strategy_from(strat_sum[i]) for i in range(len(IS))}


def evaluate(sigma):
    """Expected payoff to player 0 when both players follow sigma."""
    memo = {}

    def val(nid):
        if nid in memo:
            return memo[nid]
        n = nodes[nid]
        if n['kind'] == 'T':
            r = n['u']
        elif n['kind'] == 'C':
            r = sum(p * val(ch) for p, ch in n['children'])
        else:
            r = sum(sigma[n['iset']][a] * val(ch) for a, ch in enumerate(n['children']))
        memo[nid] = r
        return r

    return val(0)


def best_response(br, sigma):
    """Best-response value for player br against sigma (in br's own payoff units)."""
    sgn = 1.0 if br == 0 else -1.0
    N = len(nodes)
    # Reach of the opponent and chance from the root (the best responder's own factor is 1).
    w = [0.0] * N
    w[0] = 1.0
    for nid in range(N):
        n = nodes[nid]
        if n['kind'] == 'C':
            for p, ch in n['children']:
                w[ch] += w[nid] * p
        elif n['kind'] == 'P':
            for a, ch in enumerate(n['children']):
                f = 1.0 if n['player'] == br else sigma[n['iset']][a]
                w[ch] += w[nid] * f

    memo = {}
    choice = {}

    def val(nid):
        if nid in memo:
            return memo[nid]
        n = nodes[nid]
        if n['kind'] == 'T':
            r = sgn * n['u']
        elif n['kind'] == 'C':
            r = sum(p * val(ch) for p, ch in n['children'])
        elif n['player'] == br:
            r = val(n['children'][choice[n['iset']]])
        else:
            r = sum(sigma[n['iset']][a] * val(ch) for a, ch in enumerate(n['children']))
        memo[nid] = r
        return r

    # Deeper information sets first: their choices are final before shallower ones are scored.
    mine = [i for i in range(len(IS)) if IS[i]['player'] == br]
    mine.sort(key=lambda i: -IS[i]['depth'])
    for iid in mine:
        info = IS[iid]
        k = len(info['letters'])
        tot = [0.0] * k
        for nid in info['nodes']:
            for a, ch in enumerate(nodes[nid]['children']):
                tot[a] += w[nid] * val(ch)
        choice[iid] = max(range(k), key=lambda a: tot[a])
    return val(0)


def exploitability(sigma):
    b0 = best_response(0, sigma)
    b1 = best_response(1, sigma)  # player 1's payoff units
    return (b0 + b1) / 2.0, b0, b1


# ---------------------------------------------------------------- main
def main():
    t0 = time.time()
    root = build()
    assert root == 0
    print(f"tree: {len(nodes)} nodes, {len(IS)} information sets", flush=True)
    regret = [[0.0] * len(info['letters']) for info in IS]
    strat_sum = [[0.0] * len(info['letters']) for info in IS]

    for t in range(1, ITERATIONS + 1):
        walk(0, 1.0, 1.0, 1.0, 0, t, regret, strat_sum)
        walk(0, 1.0, 1.0, 1.0, 1, t, regret, strat_sum)
        if t % 1000 == 0 or t == ITERATIONS:
            sigma = average_strategy(strat_sum)
            e, b0, b1 = exploitability(sigma)
            print(f"iter {t:6d}  exploitability {e:.3e}  (br0 {b0:.3e}, br1 {b1:.3e})"
                  f"  elapsed {time.time() - t0:.0f}s", flush=True)

    sigma = average_strategy(strat_sum)
    e, b0, b1 = exploitability(sigma)
    value = evaluate(sigma)

    probs = {}
    for iid, info in enumerate(IS):
        acts = [LABEL[L] for L in info['letters']]
        probs[info['name']] = {acts[a]: repr(float(sigma[iid][a])) for a in range(len(acts))}

    doc = {
        'format': 'bluff-strategy-v1',
        'game': 'leduc',
        'producer': f'CFR+ ({ITERATIONS} iterations, alternating updates, linear averaging) '
                    f'on the full Leduc hold\'em tree',
        'probs': dict(sorted(probs.items())),
    }
    with open(OUT_PATH, 'w') as f:
        json.dump(doc, f, indent=1)
        f.write('\n')

    print(f"wrote {OUT_PATH}: {len(probs)} information sets")
    print(f"exploitability {e:.6e}  (br0 {b0:.6e}, br1 {b1:.6e})")
    print(f"value to player 0 {value:.6f}")
    print(f"elapsed {time.time() - t0:.0f}s")
    result = {
        'claim_converged': e < 1e-3,
        'claimed_exploitability': e,
        'claimed_value_p0': value,
        'confidence': 'high' if e < 1e-3 else 'medium',
        'notes': f'CFR+ with {ITERATIONS} iterations on the full game tree; exploitability computed '
                 f'exactly by best responses over the full tree; sum of probabilities per '
                 f'information set is 1 up to float rounding.',
    }
    print(json.dumps(result))


if __name__ == '__main__':
    main()
