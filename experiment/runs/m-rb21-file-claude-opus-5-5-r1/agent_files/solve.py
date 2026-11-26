import json, sys
from fractions import Fraction
sys.setrecursionlimit(100000)
g = json.load(open('game.json'))
N = {n['id']: n for n in g['nodes']}
for n in N.values():
    if n['kind'] == 'chance':
        n['p'] = [float(Fraction(o['prob'])) for o in n['outcomes']]
        n['ch'] = [o['child'] for o in n['outcomes']]
    elif n['kind'] == 'decision':
        n['ch'] = [a['child'] for a in n['actions']]
    else:
        n['u'] = float(Fraction(n['payoff_p0']))
infosets = {}
for n in N.values():
    if n['kind'] == 'decision':
        labels = [a['label'] for a in n['actions']]
        if n['infoset'] in infosets:
            assert infosets[n['infoset']][1] == labels
        else:
            infosets[n['infoset']] = (n['player'], labels)
R = {I: [0.0]*len(v[1]) for I, v in infosets.items()}
S = {I: [0.0]*len(v[1]) for I, v in infosets.items()}

def cur(I):
    r = R[I]; pos = [max(x, 0.0) for x in r]; s = sum(pos)
    return [x/s for x in pos] if s > 0 else [1.0/len(r)]*len(r)

def cfr(nid, tp, pi_opp, pi_me, w):
    # returns value for player tp
    n = N[nid]; k = n['kind']
    if k == 'terminal':
        return n['u'] if tp == 0 else -n['u']
    if k == 'chance':
        return sum(p*cfr(c, tp, pi_opp*p, pi_me, w) for p, c in zip(n['p'], n['ch']))
    I = n['infoset']; sig = cur(I)
    if n['player'] == tp:
        vs = [cfr(c, tp, pi_opp, pi_me*s, w) for s, c in zip(sig, n['ch'])]
        v = sum(s*x for s, x in zip(sig, vs))
        r = R[I]; st = S[I]
        for a in range(len(vs)):
            r[a] += pi_opp*(vs[a]-v)
            st[a] += pi_me*sig[a]
        return v
    else:
        return sum(s*cfr(c, tp, pi_opp*s, pi_me, w) for s, c in zip(sig, n['ch']) )

def avg(I):
    s = S[I]; t = sum(s)
    return [x/t for x in s] if t > 0 else [1.0/len(s)]*len(s)

def value(nid, strat):
    n = N[nid]; k = n['kind']
    if k == 'terminal': return n['u']
    if k == 'chance': return sum(p*value(c, strat) for p, c in zip(n['p'], n['ch']))
    return sum(s*value(c, strat) for s, c in zip(strat[n['infoset']], n['ch']))

def best_response(br, strat):
    # collect nodes per infoset with opponent*chance reach
    nodes_of = {}
    def walk(nid, reach):
        n = N[nid]
        if n['kind'] == 'terminal': return
        if n['kind'] == 'chance':
            for p, c in zip(n['p'], n['ch']): walk(c, reach*p)
            return
        if n['player'] == br:
            nodes_of.setdefault(n['infoset'], []).append((nid, reach))
            for c in n['ch']: walk(c, reach)
        else:
            for s, c in zip(strat[n['infoset']], n['ch']): walk(c, reach*s)
    walk(g['root'], 1.0)
    choice = {}; memo = {}
    def v(nid):  # value for br player, weighted by nothing (conditional on reaching)
        if nid in memo: return memo[nid]
        n = N[nid]; k = n['kind']
        if k == 'terminal': r = n['u'] if br == 0 else -n['u']
        elif k == 'chance': r = sum(p*v(c) for p, c in zip(n['p'], n['ch']))
        elif n['player'] == br:
            I = n['infoset']
            if I not in choice:
                tot = [0.0]*len(n['ch'])
                for (h, reach) in nodes_of.get(I, [(nid, 1.0)]):
                    for a, c in enumerate(N[h]['ch']): tot[a] += reach*v(c)
                choice[I] = max(range(len(tot)), key=lambda a: tot[a])
            r = v(n['ch'][choice[I]])
        else:
            r = sum(s*v(c) for s, c in zip(strat[n['infoset']], n['ch']))
        memo[nid] = r; return r
    return v(g['root'])

iters = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
for t in range(1, iters+1):
    for p in (0, 1):
        cfr(g['root'], p, 1.0, 1.0, 1.0)
    # DCFR(alpha=1.5, beta=0, gamma=2) discounting
    pa = t**1.5/(t**1.5+1); ga = (t/(t+1))**2
    for I in R:
        R[I] = [x*pa if x > 0 else x*0.5 for x in R[I]]
        S[I] = [x*ga for x in S[I]]
    if t % 3000 == 0 or t == iters:
        strat = {I: avg(I) for I in infosets}
        b0 = best_response(0, strat); b1 = best_response(1, strat)
        print(t, 'v0=%.6f br0=%.6f br1=%.6f expl=%.3e' % (value(g['root'], strat), b0, b1, (b0+b1)/2), flush=True)
strat = {I: avg(I) for I in infosets}
out = {"format": "bluff-strategy-v1", "game": g['name'], "producer": "Discounted CFR (alpha=1.5, beta=0, gamma=2), alternating updates, solve.py, %d iters" % iters,
       "probs": {I: {lab: repr(p) for lab, p in zip(infosets[I][1], strat[I])} for I in infosets}}
# ensure exact sums: fix last
def exact(d):
    from fractions import Fraction as F
    ks = list(d); vals = [F(d[k]) for k in ks[:-1]]
    last = 1 - sum(vals, F(0))
    assert last >= 0
    r = {k: d[k] for k in ks[:-1]}; r[ks[-1]] = str(last); return r
out['probs'] = {I: exact(d) for I, d in out['probs'].items()}
json.dump(out, open('strategy.json', 'w'), indent=1)
