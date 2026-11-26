import json, itertools
from fractions import Fraction
CARDS="JQK"; RANK={'J':0,'Q':1,'K':2}
def actions(h): return ['fold','call'] if h in('b','cb') else ['check','bet']
def term(h):
    # returns payoff to player 0 function of (c0,c1) or None
    if h in('cc','bc','cbc'):
        return lambda a,b:(1 if RANK[a]>RANK[b] else -1)*(1 if h=='cc' else 2)
    if h=='bf': return lambda a,b:1
    if h=='cbf': return lambda a,b:-1
    return None
NEXT={'c':'check','b':'bet'}
def step(h,a):
    m={'check':'c','bet':'b'}
    if a in m: return h+m[a]
    if a=='call': return h+'c'
    return h+'f'
def player(h): return len(h)%2
def hist_name(h): return h
regret={};ssum={}
def strat(k,n=2):
    r=regret.setdefault(k,[0.0]*n); p=[max(x,0) for x in r]; s=sum(p)
    return [x/s for x in p] if s>0 else [1/n]*n
def cfr(h,cards,reach):
    t=term(h)
    if t: return t(*cards)
    p=player(h); k=CARDS[cards[p]] if False else cards[p]+':'+h
    acts=actions(h); s=strat(k)
    ss=ssum.setdefault(k,[0.0]*2)
    for i in range(2): ss[i]+=reach[p]*s[i]
    u=[]
    for i,a in enumerate(acts):
        r2=list(reach); r2[p]*=s[i]
        u.append(cfr(step(h,a),cards,r2))
    sign=1 if p==0 else -1
    v=sum(s[i]*u[i] for i in range(2))
    for i in range(2): regret[k][i]+=reach[1-p]*sign*(u[i]-v)
    return v
def run(N):
    for _ in range(N):
        for a,b in itertools.permutations(CARDS,2):
            cfr('',(a,b),[1,1])
def avg():
    out={}
    for k,v in ssum.items():
        s=sum(v); out[k]=[x/s for x in v] if s>0 else [.5,.5]
    return out
def value(S):
    tot=0
    def go(h,c,pr):
        t=term(h)
        if t: return pr*t(*c)
        p=player(h);s=S[c[p]+':'+h]
        return sum(go(step(h,a),c,pr*s[i]) for i,a in enumerate(actions(h)))
    for c in itertools.permutations(CARDS,2): tot+=go('',c,1/6)
    return tot
def br(S,bp):
    # best response value for bp, via info-set-level maximization over opponent-consistent beliefs
    o=1-bp; total=0
    for mine in CARDS:
        def go(h,opp,pr):
            t=term(h)
            if t:
                return pr*(t(mine,opp) if bp==0 else -t(opp,mine))
            p=player(h)
            if p==o:
                s=S[opp+':'+h]
                return sum(go(step(h,a),opp,pr*s[i]) for i,a in enumerate(actions(h)))
            return None
        # need joint maximization across opp cards: do recursive over histories
        def rec(h,w):  # w: dict opp->prob weight
            p=player(h); acts=actions(h)
            t=term(h)
            if t: return sum(wt*(t(mine,oc) if bp==0 else -t(oc,mine)) for oc,wt in w.items())
            if p==bp: return max(rec(step(h,a),w) for a in acts)
            tot=0
            for i,a in enumerate(acts):
                tot+=rec(step(h,a),{oc:wt*S[oc+':'+h][i] for oc,wt in w.items()})
            return tot
        total+=rec('',{oc:1/6 for oc in CARDS if oc!=mine})
    return total
if __name__=="__main__":
    run(200000)
    S=avg()
    ex=(br(S,0)+br(S,1))/2
    v=value(S)
    print("expl",ex,"value",v)
    probs={k:{a:repr(S[k][i]) for i,a in enumerate(actions(k.split(':')[1]))} for k in sorted(S)}
    json.dump({"format":"bluff-strategy-v1","game":"kuhn","producer":"vanilla CFR 200k iters, average strategy","probs":probs},open("strategy.json","w"),indent=1)
    for k in sorted(S):print(k,probs[k])
