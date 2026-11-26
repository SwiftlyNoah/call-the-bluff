import json, sys
from fractions import Fraction
g = json.load(open('game.json'))
N = {n['id']: n for n in g['nodes']}
F = lambda s: float(Fraction(s))
acts = {}
for n in N.values():
    if n['kind'] == 'decision':
        acts[n['infoset']] = [a['label'] for a in n['actions']]
cum = {i: [0.0]*len(a) for i, a in acts.items()}   # regrets
ssum = {i: [0.0]*len(a) for i, a in acts.items()}
def strat(i):
    r = [max(x, 0) for x in cum[i]]; s = sum(r)
    return [x/s for x in r] if s > 0 else [1/len(r)]*len(r)
sys.setrecursionlimit(10000)
def cfr(nid, r0, r1, w):
    n = N[nid]; k = n['kind']
    if k == 'terminal': return F(n['payoff_p0'])
    if k == 'chance':
        return sum(F(o['prob'])*cfr(o['child'], r0, r1, w) for o in n['outcomes'])
    p = n['player']; i = n['infoset']; s = strat(i)
    vs = []
    for a, pr in zip(n['actions'], s):
        vs.append(cfr(a['child'], r0*pr if p == 0 else r0, r1*pr if p == 1 else r1, w))
    v = sum(pr*x for pr, x in zip(s, vs))
    mine, opp = (r0, r1) if p == 0 else (r1, r0)
    for j in range(len(s)):
        d = (vs[j]-v) if p == 0 else (v-vs[j])
        cum[i][j] = max(cum[i][j] + opp*d, 0.0)   # CFR+
        ssum[i][j] += w*mine*s[j]
    return v
def avg(i):
    s = sum(ssum[i]); return [x/s for x in ssum[i]] if s > 0 else [1/len(ssum[i])]*len(ssum[i])
def evalv(nid, sig, r=1.0):
    n = N[nid]; k = n['kind']
    if k == 'terminal': return F(n['payoff_p0'])
    if k == 'chance': return sum(F(o['prob'])*evalv(o['child'], sig) for o in n['outcomes'])
    return sum(pr*evalv(a['child'], sig) for a, pr in zip(n['actions'], sig[n['infoset']]))
def br(p, sig):
    choice = {}
    for _ in range(30):
        acc = {i: [0.0]*len(a) for i, a in acts.items()}
        def go(nid, opp):  # value for p
            n = N[nid]; k = n['kind']
            if k == 'terminal': return F(n['payoff_p0'])*(1 if p == 0 else -1)
            if k == 'chance': return sum(F(o['prob'])*go(o['child'], opp) for o in n['outcomes'])
            i = n['infoset']
            if n['player'] == p:
                vs = [go(a['child'], opp) for a in n['actions']]
                for j, x in enumerate(vs): acc[i][j] += opp*x
                c = choice.get(i, 0); return vs[c]
            return sum(pr*go(a['child'], opp*pr) for a, pr in zip(n['actions'], sig[i]))
        val = go(g['root'], 1.0)
        new = {i: max(range(len(a)), key=lambda j: acc[i][j]) for i, a in acc.items() if any(N[x]['player']==p for x in [next(nn['id'] for nn in N.values() if nn['kind']=='decision' and nn['infoset']==i)])}
        if new == choice: return val
        choice = new
    return val
def expl(sig): return (br(0, sig)+br(1, sig))/2
T = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
for t in range(1, T+1):
    cfr(g['root'], 1.0, 1.0, t)
    if t % 500 == 0:
        sig = {i: avg(i) for i in acts}; print(t, expl(sig), flush=True)
sig = {i: avg(i) for i in acts}
print('expl', expl(sig), 'value', evalv(g['root'], sig))
json.dump({"format": "bluff-strategy-v1", "game": g['name'], "producer": "CFR+ linear avg",
           "probs": {i: {a: repr(p) for a, p in zip(acts[i], sig[i])} for i in acts}}, open('strategy.json', 'w'))
