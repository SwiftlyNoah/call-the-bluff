import json, sys
from fractions import Fraction
sys.setrecursionlimit(10000)
g = json.load(open('game.json'))
N = {n['id']: n for n in g['nodes']}
F = lambda s: Fraction(s)
IS = {}  # infoset -> (player, labels, [node ids])
for n in g['nodes']:
    if n['kind'] == 'decision':
        IS.setdefault(n['infoset'], (n['player'], [a['label'] for a in n['actions']], []))[2].append(n['id'])
reg = {k: [0.0]*len(v[1]) for k, v in IS.items()}
ssum = {k: [0.0]*len(v[1]) for k, v in IS.items()}
def strat(k):
    r = [max(x, 0.0) for x in reg[k]]; s = sum(r); n = len(r)
    return [x/s for x in r] if s > 0 else [1.0/n]*n
def cfr(nid, p, reach, t):
    n = N[nid]
    if n['kind'] == 'terminal': return float(F(n['payoff_p0']))
    if n['kind'] == 'chance':
        return sum(float(F(o['prob']))*cfr(o['child'], p, [reach[0], reach[1], reach[2]*float(F(o['prob']))], t) for o in n['outcomes'])
    k = n['infoset']; pl = n['player']; s = strat(k)
    vals = []
    for i, a in enumerate(n['actions']):
        r = list(reach); r[pl] *= s[i]
        vals.append(cfr(a['child'], p, r, t))
    v = sum(si*vi for si, vi in zip(s, vals))
    opp = reach[1-pl]*reach[2]
    sign = 1 if pl == 0 else -1
    for i in range(len(vals)):
        reg[k][i] = max(0.0, reg[k][i] + opp*sign*(vals[i]-v))
        ssum[k][i] += t*reach[pl]*s[i]   # linear averaging
    return v
T = 20000
for t in range(1, T+1):
    cfr(g['root'], 0, [1.0, 1.0, 1.0], t)
avg = {}
for k in IS:
    s = sum(ssum[k]); n = len(ssum[k])
    avg[k] = [x/s for x in ssum[k]] if s > 0 else [1.0/n]*n
def value(nid):
    n = N[nid]
    if n['kind'] == 'terminal': return float(F(n['payoff_p0']))
    if n['kind'] == 'chance': return sum(float(F(o['prob']))*value(o['child']) for o in n['outcomes'])
    s = avg[n['infoset']]
    return sum(s[i]*value(a['child']) for i, a in enumerate(n['actions']))
def br(player):
    sg = 1 if player == 0 else -1
    memo = {}
    def V(nid):
        n = N[nid]
        if n['kind'] == 'terminal': return sg*float(F(n['payoff_p0']))
        if n['kind'] == 'chance': return sum(float(F(o['prob']))*V(o['child']) for o in n['outcomes'])
        if n['player'] == player:
            return best(n['infoset'])[1][0] if False else choose(n['infoset'], nid)
        s = avg[n['infoset']]
        return sum(s[i]*V(a['child']) for i, a in enumerate(n['actions']))
    def weights(nid, w, out):
        n = N[nid]
        if n['kind'] == 'terminal': return
        if n['kind'] == 'chance':
            for o in n['outcomes']: weights(o['child'], w*float(F(o['prob'])), out)
            return
        if n['player'] == player:
            out.setdefault(n['infoset'], {})[nid] = w
            for a in n['actions']: weights(a['child'], w, out)
        else:
            s = avg[n['infoset']]
            for i, a in enumerate(n['actions']): weights(a['child'], w*s[i], out)
    W = {}; weights(g['root'], 1.0, W)
    act = {}
    def best(k):
        if k not in act:
            tot = [0.0]*len(IS[k][1])
            for nid, w in W[k].items():
                for i, a in enumerate(N[nid]['actions']): tot[i] += w*V(a['child'])
            act[k] = max(range(len(tot)), key=lambda i: tot[i])
        return act[k]
    def choose(k, nid):
        return V(N[nid]['actions'][best(k)]['child'])
    return V(g['root'])
e0, e1 = br(0), br(1)
expl = (e0+e1)/2
v = value(g['root'])
print('BR0', e0, 'BR1', e1, 'expl', expl, 'value', v)
json.dump({"format": "bluff-strategy-v1", "game": g['name'], "producer": "CFR+ (linear averaging), %d iterations" % T,
           "probs": {k: {l: repr(avg[k][i]) for i, l in enumerate(IS[k][1])} for k in IS}}, open('strategy.json', 'w'), indent=1)
