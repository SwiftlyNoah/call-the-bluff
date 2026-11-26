import json, sys
from fractions import Fraction
import numpy as np

def num(s): return float(Fraction(s))

g = json.load(open('game.json'))
nodes = {n['id']: n for n in g['nodes']}
root = g['root']
infosets = {}  # name -> (player, labels)
for n in g['nodes']:
    if n['kind'] == 'decision':
        labels = [a['label'] for a in n['actions']]
        if n['infoset'] in infosets:
            assert infosets[n['infoset']] == (n['player'], labels)
        infosets[n['infoset']] = (n['player'], labels)
names = sorted(infosets)
idx = {k: i for i, k in enumerate(names)}
# compact arrays
kind, player, iset, children, probs, payoff = {}, {}, {}, {}, {}, {}
for n in g['nodes']:
    i = n['id']; kind[i] = n['kind']
    if n['kind'] == 'chance':
        children[i] = [o['child'] for o in n['outcomes']]; probs[i] = [num(o['prob']) for o in n['outcomes']]
    elif n['kind'] == 'decision':
        player[i] = n['player']; iset[i] = idx[n['infoset']]; children[i] = [a['child'] for a in n['actions']]
    else:
        payoff[i] = num(n['payoff_p0'])
nA = [len(infosets[k][1]) for k in names]
regret = [np.zeros(k) for k in nA]
ssum = [np.zeros(k) for k in nA]

def cur(I):
    r = np.maximum(regret[I], 0); s = r.sum()
    return r / s if s > 0 else np.full(nA[I], 1.0 / nA[I])

sys.setrecursionlimit(100000)
def cfr(h, upd, r0, r1, rc, w, strat):
    # returns value for player 0
    k = kind[h]
    if k == 'terminal': return payoff[h]
    if k == 'chance':
        return sum(p * cfr(c, upd, r0, r1, rc * p, w, strat) for c, p in zip(children[h], probs[h]))
    I = iset[h]; pl = player[h]; s = strat[I]
    vals = np.empty(nA[I])
    for a, c in enumerate(children[h]):
        if pl == 0: vals[a] = cfr(c, upd, r0 * s[a], r1, rc, w, strat)
        else: vals[a] = cfr(c, upd, r0, r1 * s[a], rc, w, strat)
    v = float(s @ vals)
    if pl == upd:
        sign = 1 if pl == 0 else -1
        opp = (r1 if pl == 0 else r0) * rc
        regret[I] += sign * opp * (vals - v)
        ssum[I] += w * (r0 if pl == 0 else r1) * s
    return v

def avg():
    out = []
    for I in range(len(names)):
        s = ssum[I].sum()
        out.append(ssum[I] / s if s > 0 else np.full(nA[I], 1.0 / nA[I]))
    return out

def ev(h, strat):
    k = kind[h]
    if k == 'terminal': return payoff[h]
    if k == 'chance': return sum(p * ev(c, strat) for c, p in zip(children[h], probs[h]))
    s = strat[iset[h]]
    return sum(s[a] * ev(c, strat) for a, c in enumerate(children[h]))

# nodes per infoset for best response
inodes = {}
for n in g['nodes']:
    if n['kind'] == 'decision': inodes.setdefault(iset[n['id']], []).append(n['id'])
def best_response(br, strat):
    # opponent+chance reach for each node
    reach = {}
    def rr(h, r):
        reach[h] = r
        if kind[h] == 'chance':
            for c, p in zip(children[h], probs[h]): rr(c, r * p)
        elif kind[h] == 'decision':
            s = strat[iset[h]]
            for a, c in enumerate(children[h]): rr(c, r * (s[a] if player[h] != br else 1.0))
    rr(root, 1.0)
    choice = {}
    sign = 1 if br == 0 else -1
    def val(h):  # value for br player, unweighted
        k = kind[h]
        if k == 'terminal': return sign * payoff[h]
        if k == 'chance': return sum(p * val(c) for c, p in zip(children[h], probs[h]))
        I = iset[h]
        if player[h] == br:
            return val(children[h][decide(I)])
        s = strat[I]
        return sum(s[a] * val(c) for a, c in enumerate(children[h]) if s[a] > 0)
    def decide(I):
        if I in choice: return choice[I]
        choice[I] = None  # cycle guard
        tot = np.zeros(nA[I])
        for h in inodes[I]:
            if reach[h] > 0:
                for a, c in enumerate(children[h]): tot[a] += reach[h] * val(c)
        choice[I] = int(np.argmax(tot)); return choice[I]
    return val(root)

def exploit(strat):
    b0 = best_response(0, strat); b1 = best_response(1, strat)
    return (b0 + b1) / 2, b0, b1

T = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
for t in range(1, T + 1):
    for p in (0, 1):
        strat = [cur(I) for I in range(len(names))]
        cfr(root, p, 1.0, 1.0, 1.0, t, strat)  # linear averaging (CFR+)
        for I in range(len(names)):
            if infosets[names[I]][0] == p: np.maximum(regret[I], 0, out=regret[I])
    if t % 250 == 0 or t == T:
        a = avg(); e = exploit(a)
        print(t, 'exploit', e[0], 'v0', ev(root, a), flush=True)
a = avg()
json.dump({"format": "bluff-strategy-v1", "game": g['name'], "producer": f"CFR+ (alternating, linear avg) {T} iters, solve.py",
           "probs": {names[I]: {l: repr(float(a[I][j])) for j, l in enumerate(infosets[names[I]][1])} for I in range(len(names))}},
          open('strategy.json', 'w'), indent=1)
e = exploit(a); print('final exploit', e, 'v0', ev(root, a))
