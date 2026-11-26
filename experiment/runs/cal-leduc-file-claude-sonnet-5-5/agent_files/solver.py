import json, sys
from fractions import Fraction
sys.setrecursionlimit(10000)
g = json.load(open('game.json'))
N = {n['id']: n for n in g['nodes']}
fr = lambda s: float(Fraction(s))
acts = {}; owner = {}
for n in g['nodes']:
    if n['kind'] == 'decision':
        acts[n['infoset']] = [a['label'] for a in n['actions']]; owner[n['infoset']] = n['player']
reg = {i: [0.0]*len(a) for i, a in acts.items()}
ssum = {i: [0.0]*len(a) for i, a in acts.items()}
def strat(i):
    r = [max(x, 0) for x in reg[i]]; s = sum(r)
    return [x/s for x in r] if s > 0 else [1/len(r)]*len(r)
def avg(i):
    s = sum(ssum[i]); return [x/s for x in ssum[i]] if s > 0 else [1/len(ssum[i])]*len(ssum[i])
def cfr(id, p, r0, r1, w):
    n = N[id]; k = n['kind']
    if k == 'terminal': return fr(n['payoff_p0'])
    if k == 'chance':
        return sum(cfr(o['child'], p, r0, r1, w) * fr(o['prob']) for o in n['outcomes'])
    i = n['infoset']; pl = n['player']; s = strat(i)
    vals = [cfr(a['child'], p, r0*s[j] if pl == 0 else r0, r1*s[j] if pl == 1 else r1, w) for j, a in enumerate(n['actions'])]
    v = sum(x*y for x, y in zip(s, vals))
    if pl == p:
        opp = r1 if pl == 0 else r0
        for j in range(len(s)):
            d = (vals[j]-v) if pl == 0 else (v-vals[j])
            reg[i][j] = max(0.0, reg[i][j] + opp*d)
        me = r0 if pl == 0 else r1
        for j in range(len(s)): ssum[i][j] += w*me*s[j]
    return v
def ev(id, sf):
    n = N[id]; k = n['kind']
    if k == 'terminal': return fr(n['payoff_p0'])
    if k == 'chance': return sum(ev(o['child'], sf)*fr(o['prob']) for o in n['outcomes'])
    s = sf[n['infoset']]
    return sum(s[j]*ev(a['child'], sf) for j, a in enumerate(n['actions']))
def br(sf, p):
    choice = {}
    for _ in range(12):
        av = {i: [0.0]*len(a) for i, a in acts.items() if owner[i] == p}
        def go(id, opp):
            n = N[id]; k = n['kind']
            if k == 'terminal': return fr(n['payoff_p0'])*(1 if p == 0 else -1)
            if k == 'chance': return sum(go(o['child'], opp*fr(o['prob'])) for o in n['outcomes'])
            i = n['infoset']
            if n['player'] == p:
                vs = [go(a['child'], opp) for a in n['actions']]
                for j, x in enumerate(vs): av[i][j] += x*opp if False else 0
                return vs[choice[i]] if i in choice else vs[0]
            s = sf[i]
            return sum(s[j]*go(a['child'], opp) for j, a in enumerate(n['actions']))
        # need unnormalized action values: weight by opp reach (incl chance)
        def go2(id, opp):
            n = N[id]; k = n['kind']
            if k == 'terminal': return fr(n['payoff_p0'])*(1 if p == 0 else -1)
            if k == 'chance': return sum(go2(o['child'], opp*fr(o['prob']))*fr(o['prob']) for o in n['outcomes'])
            i = n['infoset']
            if n['player'] == p:
                vs = [go2(a['child'], opp) for a in n['actions']]
                for j, x in enumerate(vs): av[i][j] += x*opp
                return vs[choice[i]] if i in choice else vs[0]
            s = sf[i]
            return sum(s[j]*go2(a['child'], opp*s[j]) for j, a in enumerate(n['actions']))
        val = go2(g['root'], 1.0)
        # go2 for chance: recursion multiplies prob into opp and into value -> value is expectation; av weights opp*chance, fine but value weights prob twice? no: value = sum prob*child, child values unweighted. ok
        new = {i: max(range(len(v)), key=lambda j: v[j]) for i, v in av.items()}
        if new == choice: break
        choice = new
    return val
def expl():
    sf = {i: avg(i) for i in acts}
    b0 = br(sf, 0); b1 = br(sf, 1)
    return (b0+b1)/2, ev(g['root'], sf), sf
T = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
for t in range(1, T+1):
    for p in (0, 1): cfr(g['root'], p, 1.0, 1.0, t*t)
    if t % 500 == 0:
        e, v, _ = expl(); print(t, e, v, flush=True)
e, v, sf = expl()
print('final', e, v)
json.dump({"format": "bluff-strategy-v1", "game": g['name'], "producer": "CFR+ linear avg",
           "probs": {i: {a: repr(x) for a, x in zip(acts[i], sf[i])} for i in acts}}, open('strategy.json', 'w'))
