import json, itertools, sys
CARDS = ['Js','Jh','Qs','Qh','Ks','Kh']
RANK = {'J':0,'Q':1,'K':2}
DEALS = [d for d in itertools.permutations(CARDS,3)]  # (c0,c1,pub)

def legal(r):
    # r: actions of current round string
    raises = r.count('r')
    facing = len(r)>0 and r[-1]=='r'
    acts = []
    if facing: acts.append('f')
    acts.append('c')
    if raises < 2: acts.append('r')
    return acts

def round_over(r):
    return len(r)>=2 and r[-1]=='c' or (len(r)>=1 and r[-1]=='f')

def contribs(h1, h2):
    c = [1,1]
    for rnd, hs, size in ((0,h1,2),(1,h2,4)):
        p = 0; owe = 0
        for a in (hs or ""):
            if a=='c': c[p]+=owe; owe=0
            elif a=='r': c[p]+=owe+size; owe=size
            p ^= 1
    return c

def payoff0(deal, h1, h2):
    c = contribs(h1, h2)
    hs = h2 if h2 is not None else h1
    if hs and hs[-1]=='f':
        folder = (len(hs)-1)%2
        return c[1] if folder==1 else -c[0]
    r0, r1, rp = RANK[deal[0][0]], RANK[deal[1][0]], RANK[deal[2][0]]
    s0 = (r0==rp, r0); s1 = (r1==rp, r1)
    if s0>s1: return c[1]
    if s1>s0: return -c[0]
    return 0

# Build tree nodes per (h1, h2) public history
def iskey(deal, p, h1, h2):
    if h2 is None: return f"{deal[p]}:{h1}"
    return f"{deal[p]}|{deal[2]}:{h1}/{h2}"

def state(h1,h2):
    """returns ('term',) or ('act', player, actions)"""
    if h2 is None:
        if h1 and h1[-1]=='f': return ('term',)
        if round_over(h1): return ('next',)
        return ('act', len(h1)%2, legal(h1))
    if h2 and h2[-1]=='f': return ('term',)
    if round_over(h2): return ('term',)
    return ('act', len(h2)%2, legal(h2))

def child(h1,h2,a):
    return (h1+a, h2) if h2 is None else (h1, h2+a)

regret = {}; strat_sum = {}; ISACTS = {}
def strategy(k, acts):
    r = regret.get(k)
    if r is None:
        return [1/len(acts)]*len(acts)
    pos = [max(x,0) for x in r]; s = sum(pos)
    return [x/s for x in pos] if s>0 else [1/len(acts)]*len(acts)

def cfr(deal, h1, h2, i, pi_i, pi_o, it):
    st = state(h1,h2)
    if st[0]=='term':
        v = payoff0(deal,h1,h2); return v if i==0 else -v
    if st[0]=='next': return cfr(deal, h1, '', i, pi_i, pi_o, it)
    _, p, acts = st
    k = iskey(deal,p,h1,h2); ISACTS[k] = acts
    sig = strategy(k, acts)
    if p == i:
        vs = [cfr(deal, *child(h1,h2,a), i, pi_i*s, pi_o, it) for a,s in zip(acts,sig)]
        ev = sum(s*v for s,v in zip(sig,vs))
        r = inst.setdefault(k,[0.0]*len(acts))
        for j in range(len(acts)):
            r[j] += pi_o*(vs[j]-ev)
        ss = strat_sum.setdefault(k,[0.0]*len(acts))
        for j in range(len(acts)): ss[j] += it*pi_i*sig[j]  # linear averaging
        return ev
    else:
        ev = 0.0
        for a,s in zip(acts,sig):
            if s>0: ev += s*cfr(deal, *child(h1,h2,a), i, pi_i, pi_o*s, it)
        return ev

def avg(k):
    ss = strat_sum.get(k); acts = ISACTS[k]
    if ss is None or sum(ss)<=0: return [1/len(acts)]*len(acts)
    t = sum(ss); return [x/t for x in ss]

def br(h1, h2, i, weights, S):
    """weights: dict deal->opp reach*chance. returns dict deal->value for player i."""
    st = state(h1,h2)
    if st[0]=='term':
        return {d:(payoff0(d,h1,h2) if i==0 else -payoff0(d,h1,h2)) for d in weights}
    if st[0]=='next': return br(h1,'',i,weights,S)
    _, p, acts = st
    if p == i:
        cv = [br(*child(h1,h2,a), i, weights, S) for a in acts]
        groups = {}
        for d in weights: groups.setdefault(iskey(d,p,h1,h2),[]).append(d)
        out = {}
        for k, ds in groups.items():
            best = max(range(len(acts)), key=lambda j: sum(weights[d]*cv[j][d] for d in ds))
            for d in ds: out[d] = cv[best][d]
        return out
    else:
        out = {d:0.0 for d in weights}
        for j,a in enumerate(acts):
            w2 = {d: weights[d]*S[iskey(d,p,h1,h2)][j] for d in weights}
            cv = br(*child(h1,h2,a), i, w2, S)
            for d in weights: out[d] += S[iskey(d,p,h1,h2)][j]*cv[d]
        return out

def ev(h1,h2,d,S):
    st = state(h1,h2)
    if st[0]=='term': return payoff0(d,h1,h2)
    if st[0]=='next': return ev(h1,'',d,S)
    _,p,acts = st; s = S[iskey(d,p,h1,h2)]
    return sum(s[j]*ev(*child(h1,h2,a),d,S) for j,a in enumerate(acts))

def evaluate(S):
    w = {d:1/len(DEALS) for d in DEALS}
    b0 = sum(w[d]*v for d,v in br('',None,0,w,S).items())
    b1 = sum(w[d]*v for d,v in br('',None,1,w,S).items())
    v0 = sum(ev('',None,d,S) for d in DEALS)/len(DEALS)
    return b0, b1, (b0+b1)/2, v0

N = int(sys.argv[1]) if len(sys.argv)>1 else 2000
for it in range(1,N+1):
    for i in (0,1):
        inst = {}
        for d in DEALS: cfr(d,'',None,i,1.0,1.0,it)
        for k,r in inst.items():
            R = regret.setdefault(k,[0.0]*len(r))
            for j in range(len(r)): R[j] = max(R[j]+r[j], 0.0)  # CFR+
    if it % 500 == 0 or it==N:
        S = {k:avg(k) for k in ISACTS}
        b0,b1,ex,v0 = evaluate(S)
        print(it, len(ISACTS), f"br0={b0:.6f} br1={b1:.6f} expl={ex:.6g} v0={v0:.6f}", flush=True)

S = {k:avg(k) for k in ISACTS}
L = {'f':'fold','c':'call','r':'raise'}
probs = {k:{L[a]:repr(p) for a,p in zip(ISACTS[k],S[k])} for k in sorted(ISACTS)}
json.dump({"format":"bluff-strategy-v1","game":"leduc","producer":f"CFR+ (alternating, linear averaging), {N} iterations, full traversal","probs":probs}, open('strategy.json','w'), indent=1)
