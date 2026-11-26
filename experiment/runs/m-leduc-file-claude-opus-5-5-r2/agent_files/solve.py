"""Discounted CFR (DCFR) solver for games in bluff-game-v1 format. Usage: python3 solve.py [iterations]"""
import json, sys
from fractions import Fraction

def num(s): return float(Fraction(s))

def load(path):
    g = json.load(open(path))
    nodes = {x['id']: x for x in g['nodes']}
    infosets = {}  # name -> (player, [labels])
    for x in g['nodes']:
        if x['kind'] == 'decision':
            labels = [a['label'] for a in x['actions']]
            prev = infosets.setdefault(x['infoset'], (x['player'], labels))
            assert prev == (x['player'], labels), x['infoset']
    return g, nodes, infosets


def run(iters):
    g, nodes, infosets = load('game.json')
    regret = {I: [0.0]*len(a) for I, (p, a) in infosets.items()}
    avg = {I: [0.0]*len(a) for I, (p, a) in infosets.items()}
    T = {}
    for i, x in nodes.items():
        k = x['kind']
        if k == 'terminal': T[i] = (0, num(x['payoff_p0']))
        elif k == 'chance': T[i] = (1, [(num(o['prob']), o['child']) for o in x['outcomes']])
        else: T[i] = (2, x['player'], x['infoset'], [a['child'] for a in x['actions']])
    def strat(I):
        r = regret[I]; s = sum(v for v in r if v > 0)
        return [max(v, 0.0)/s for v in r] if s > 0 else [1.0/len(r)]*len(r)
    # cf: reach of everything except updater (chance * opponent); own: reach of non-updater player only
    def cfr(i, upd, cf, own, w):
        t = T[i]
        if t[0] == 0: return t[1] if upd == 0 else -t[1]
        if t[0] == 1: return sum(p*cfr(c, upd, cf*p, own, w) for p, c in t[1])
        _, pl, I, ch = t
        s = strat(I)
        if pl == upd:
            vals = [cfr(c, upd, cf, own, w) for c in ch]
            v = sum(s[a]*vals[a] for a in range(len(ch)))
            r = regret[I]
            for a in range(len(ch)):
                r[a] = r[a] + cf*(vals[a]-v)
            return v
        av = avg[I]
        for a in range(len(ch)): av[a] += w*own*s[a]
        tot = 0.0
        for a, c in enumerate(ch):
            if s[a] > 0: tot += s[a]*cfr(c, upd, cf*s[a], own*s[a], w)
        return tot
    root = g['root']
    for t in range(1, iters+1):
        # alternating updates; average of player p accumulated during the other player's pass
        # using the strategy current at that time (linear weighting, CFR+)
        # DCFR(alpha=1.5, beta=0, gamma=2): discount accumulated regrets and average
        if t > 1:
            pw = (t-1)**1.5; pos = pw/(pw+1); neg = 0.5; ad = ((t-1)/t)**2
            for I, r in regret.items():
                for a in range(len(r)): r[a] *= pos if r[a] > 0 else neg
            for I, av in avg.items():
                for a in range(len(av)): av[a] *= ad
        for upd in (0, 1):
            cfr(root, upd, 1.0, 1.0, 1.0)
        if t % 500 == 0 or t == iters:
            sig = finalize(avg, infosets)
            e, v = evaluate(T, root, infosets, sig)
            print(f"iter {t}: exploitability {e:.6g}  value_p0 {v:.6g}", flush=True)
    return g, infosets, finalize(avg, infosets), T, root

def finalize(avg, infosets):
    out = {}
    for I, (p, labels) in infosets.items():
        a = avg[I]; s = sum(a)
        out[I] = [x/s for x in a] if s > 0 else [1.0/len(a)]*len(a)
    return out

def evaluate(T, root, infosets, sig):
    # expected value
    def ev(i):
        t = T[i]
        if t[0] == 0: return t[1]
        if t[0] == 1: return sum(p*ev(c) for p, c in t[1])
        s = sig[t[2]]
        return sum(s[a]*ev(c) for a, c in enumerate(t[3]))
    v0 = ev(root)
    # best response for player br: collect for each infoset its nodes with opp*chance reach
    def br_value(br):
        reach = {}  # node -> reach of chance*opponent
        members = {}
        stack = [(root, 1.0)]
        while stack:
            i, r = stack.pop()
            t = T[i]; reach[i] = r
            if t[0] == 0: continue
            if t[0] == 1:
                for p, c in t[1]: stack.append((c, r*p))
            elif t[1] == br:
                members.setdefault(t[2], []).append(i)
                for c in t[3]: stack.append((c, r))
            else:
                s = sig[t[2]]
                for a, c in enumerate(t[3]): stack.append((c, r*s[a]))
        choice = {}
        memo = {}
        sys.setrecursionlimit(100000)
        def val(i):  # value to br (p0 payoff sign-adjusted), not reach-weighted
            if i in memo: return memo[i]
            t = T[i]
            if t[0] == 0: v = t[1] if br == 0 else -t[1]
            elif t[0] == 1: v = sum(p*val(c) for p, c in t[1])
            elif t[1] == br:
                v = val(t[3][decide(t[2])])
            else:
                s = sig[t[2]]
                v = sum(s[a]*val(c) for a, c in enumerate(t[3]))
            memo[i] = v
            return v
        def decide(I):
            if I in choice: return choice[I]
            n = len(infosets[I][1])
            tot = [0.0]*n
            for i in members[I]:
                for a, c in enumerate(T[i][3]): tot[a] += reach[i]*val(c)
            choice[I] = max(range(n), key=lambda a: tot[a])
            return choice[I]
        return val(root)
    b0, b1 = br_value(0), br_value(1)
    return (b0 + b1)/2, v0

if __name__ == '__main__':
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    sys.setrecursionlimit(100000)
    g, infosets, sig, T, root = run(iters)
    e, v = evaluate(T, root, infosets, sig)
    probs = {I: {lab: repr(sig[I][a]) for a, lab in enumerate(labels)} for I, (p, labels) in infosets.items()}
    json.dump({"format": "bluff-strategy-v1", "game": g['name'],
               "producer": f"Discounted CFR (alpha=1.5, beta=0, gamma=2, alternating updates), {iters} iterations; exploitability {e:.3g}",
               "probs": probs}, open('strategy.json', 'w'), indent=1)
    print(f"final exploitability {e!r} value_p0 {v!r}")
