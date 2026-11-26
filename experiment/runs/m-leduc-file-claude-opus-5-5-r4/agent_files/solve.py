"""CFR+ (alternating updates, linear averaging) solver for bluff-game-v1 trees."""
import json, sys
from fractions import Fraction
sys.setrecursionlimit(100000)

def num(s): return float(Fraction(s))

game = json.load(open('game.json'))
nodes = {n['id']: n for n in game['nodes']}
root = game['root']

# flatten into arrays in preorder
order = []
stack = [root]
while stack:
    i = stack.pop(); order.append(i); n = nodes[i]
    if n['kind'] == 'chance': stack.extend(o['child'] for o in n['outcomes'])
    elif n['kind'] == 'decision': stack.extend(a['child'] for a in n['actions'])

infosets = {}  # name -> (player, labels)
for n in game['nodes']:
    if n['kind'] == 'decision':
        labels = [a['label'] for a in n['actions']]
        if n['infoset'] in infosets:
            assert infosets[n['infoset']] == (n['player'], labels)
        else:
            infosets[n['infoset']] = (n['player'], labels)
I = {k: len(v[1]) for k, v in infosets.items()}
regret = {k: [0.0]*I[k] for k in infosets}
avg = {k: [0.0]*I[k] for k in infosets}
payoff = {i: num(n['payoff_p0']) for i, n in nodes.items() if n['kind'] == 'terminal'}
chprob = {i: [(num(o['prob']), o['child']) for o in n['outcomes']] for i, n in nodes.items() if n['kind'] == 'chance'}

def cur(k):
    r = regret[k]; s = sum(x for x in r if x > 0)
    return [x/s if x > 0 else 0.0 for x in r] if s > 0 else [1.0/len(r)]*len(r)


snap = {}

def cfr(i, p, rp, ro, w):
    """Value for player p. rp: p's reach, ro: opponent*chance reach."""
    n = nodes[i]; kind = n['kind']
    if kind == 'terminal':
        return payoff[i] if p == 0 else -payoff[i]
    if kind == 'chance':
        return sum(pr * cfr(c, p, rp, ro*pr, w) for pr, c in chprob[i])
    k = n['infoset']; acts = n['actions']
    if n['player'] == p:
        if k not in snap: snap[k] = (rp, cur(k))
        s = snap[k][1]
        vals = [cfr(a['child'], p, rp*s[j], ro, w) for j, a in enumerate(acts)]
        v = sum(s[j]*vals[j] for j in range(len(acts)))
        r = regret[k]
        for j in range(len(acts)):
            r[j] += ro*(vals[j]-v)
        return v
    if ro == 0.0: return 0.0
    s = cur(k)
    return sum(s[j]*cfr(a['child'], p, rp, ro*s[j], w) for j, a in enumerate(acts) if s[j] > 0)

def avg_strategy():
    out = {}
    for k, a in avg.items():
        t = sum(a); out[k] = [x/t for x in a] if t > 0 else [1.0/len(a)]*len(a)
    return out

def evaluate(strat):
    """Return (value_p0, br_value_p0, br_value_p1)."""
    def ev(i):
        n = nodes[i]
        if n['kind'] == 'terminal': return payoff[i]
        if n['kind'] == 'chance': return sum(pr*ev(c) for pr, c in chprob[i])
        s = strat[n['infoset']]
        return sum(s[j]*ev(a['child']) for j, a in enumerate(n['actions']))
    def br(p):
        # collect, per infoset of p, its nodes with opponent*chance reach
        reach = {}
        nodes_of = {}
        def walk(i, ro):
            n = nodes[i]
            if n['kind'] == 'terminal': return
            if n['kind'] == 'chance':
                for pr, c in chprob[i]: walk(c, ro*pr)
                return
            if n['player'] == p:
                nodes_of.setdefault(n['infoset'], []).append(i); reach[i] = ro
                for a in n['actions']: walk(a['child'], ro)
            else:
                s = strat[n['infoset']]
                for j, a in enumerate(n['actions']): walk(a['child'], ro*s[j])
        walk(root, 1.0)
        choice = {}; memo = {}
        def val(i):  # value for p (not reach-weighted) under BR
            if i in memo: return memo[i]
            n = nodes[i]
            if n['kind'] == 'terminal': v = payoff[i] if p == 0 else -payoff[i]
            elif n['kind'] == 'chance': v = sum(pr*val(c) for pr, c in chprob[i])
            elif n['player'] == p:
                k = n['infoset']
                if k not in choice:
                    tot = [0.0]*I[k]
                    for m in nodes_of[k]:
                        for j, a in enumerate(nodes[m]['actions']):
                            tot[j] += reach[m]*val(a['child'])
                    choice[k] = max(range(I[k]), key=lambda j: tot[j])
                v = val(n['actions'][choice[k]]['child'])
            else:
                s = strat[n['infoset']]
                v = sum(s[j]*val(a['child']) for j, a in enumerate(n['actions']) if s[j] > 0)
            memo[i] = v; return v
        return val(root)
    return ev(root), br(0), br(1)

if __name__ == '__main__':
    import time
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    t0 = time.time()
    for t in range(1, iters+1):
        for p in (0, 1):
            snap.clear()
            cfr(root, p, 1.0, 1.0, float(t))
            for k, (rp, st_) in snap.items():
                a_ = avg[k]
                for j in range(len(a_)): a_[j] += t*rp*st_[j]
            for k, (pl, _) in infosets.items():
                if pl == p: regret[k] = [max(x, 0.0) for x in regret[k]]
        if t % 250 == 0 or t == iters:
            v, b0, b1 = evaluate(avg_strategy())
            print(f"iter {t} {time.time()-t0:.0f}s v0={v:.6f} expl={(b0+b1)/2:.3e}", flush=True)
    st = avg_strategy()
    v, b0, b1 = evaluate(st)
    json.dump({"format": "bluff-strategy-v1", "game": game.get('name', 'leduc'),
               "producer": f"CFR+ (alternating, linear averaging), {iters} iterations; exploitability {(b0+b1)/2:.3e}",
               "probs": {k: {lab: repr(st[k][j]) for j, lab in enumerate(infosets[k][1])} for k in infosets}},
              open('strategy.json', 'w'), indent=1)
    print(json.dumps({"value_p0": v, "br0": b0, "br1": b1, "expl": (b0+b1)/2, "n_infosets": len(infosets)}))
