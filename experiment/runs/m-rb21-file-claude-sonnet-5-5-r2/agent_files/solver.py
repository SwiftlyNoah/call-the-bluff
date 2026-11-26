import json, sys, time
from fractions import Fraction
sys.setrecursionlimit(10000)
g=json.load(open('game.json'))
N={n['id']:n for n in g['nodes']}
def F(s): return float(Fraction(s))
for n in N.values():
    if n['kind']=='terminal': n['u']=F(n['payoff_p0'])
    elif n['kind']=='chance': n['o']=[(F(o['prob']),o['child']) for o in n['outcomes']]
IS={}
for n in N.values():
    if n['kind']=='decision':
        IS.setdefault(n['infoset'],{'labels':[a['label'] for a in n['actions']],'player':n['player']})
for v in IS.values():
    k=len(v['labels']); v['reg']=[0.0]*k; v['sum']=[0.0]*k
def strat(v):
    r=[max(x,0) for x in v['reg']]; s=sum(r); k=len(r)
    return [x/s for x in r] if s>0 else [1/k]*k
def cfr(i,p0,p1,t):
    n=N[i]; k=n['kind']
    if k=='terminal': return n['u']
    if k=='chance': return sum(p*cfr(c,p0*p,p1*p,t) for p,c in n['o'])
    v=IS[n['infoset']]; s=strat(v); pl=v['player']
    us=[cfr(a['child'],p0*s[j] if pl==0 else p0,p1*s[j] if pl==1 else p1,t) for j,a in enumerate(n['actions'])]
    u=sum(x*y for x,y in zip(s,us))
    mine,opp=(p0,p1) if pl==0 else (p1,p0)
    for j in range(len(s)):
        d=us[j]-u
        v['reg'][j]=max(0.0,v['reg'][j]+opp*(d if pl==0 else -d))
        v['sum'][j]+=t*mine*s[j]
    return u
def avg(v):
    s=sum(v['sum']); k=len(v['sum'])
    return [x/s for x in v['sum']] if s>0 else [1/k]*k
AV=None
def br(i,pl,reach):
    # returns value to pl, given opp avg strategy; infoset-based BR via counterfactual values
    pass
def expl():
    A={k:avg(v) for k,v in IS.items()}
    def val(i):
        n=N[i];k=n['kind']
        if k=='terminal':return n['u']
        if k=='chance':return sum(p*val(c) for p,c in n['o'])
        s=A[n['infoset']];return sum(s[j]*val(a['child']) for j,a in enumerate(n['actions']))
    res=[]
    for pl in (0,1):
        cv={}
        def walk(i,q): # q: opp+chance reach; accumulate cv per infoset/action
            n=N[i];k=n['kind']
            if k=='terminal': return
            if k=='chance':
                for p,c in n['o']: walk(c,q*p)
                return
            if n['player']==pl:
                d=cv.setdefault(n['infoset'],[0.0]*len(n['actions']))
                for j,a in enumerate(n['actions']):
                    d[j]+=q*0 # placeholder
                    walk(a['child'],q)
            else:
                s=A[n['infoset']]
                for j,a in enumerate(n['actions']): walk(a['child'],q*s[j])
        # simpler: compute BR by recursion with infoset choices fixed bottom-up
        order=[]
        def collect(i,q,acc):
            pass
        # per-infoset BR via two-pass: value of action = sum over nodes of q * val under BR
        # do iterative: compute infoset depth order by memoized recursion
        choice={}
        depth={}
        def dfs(i,d):
            n=N[i];k=n['kind']
            if k=='terminal':return
            if k=='chance':
                for p,c in n['o']:dfs(c,d)
                return
            if n['player']==pl: depth[n['infoset']]=max(depth.get(n['infoset'],0),d)
            for a in n['actions']:dfs(a['child'],d+1)
        dfs(g['root'],0)
        sg=1 if pl==0 else -1
        def ev(i,q,store):
            n=N[i];k=n['kind']
            if k=='terminal':return sg*n['u']*q
            if k=='chance':return sum(ev(c,q*p,store) for p,c in n['o'])
            if n['player']==pl:
                I=n['infoset']
                if store is not None:
                    d=store.setdefault(I,[0.0]*len(n['actions']))
                    xs=[ev(a['child'],q,store) for a in n['actions']]
                    for j,x in enumerate(xs): d[j]+=x
                    return xs[choice[I]]
                return ev(n['actions'][choice[n['infoset']]]['child'],q,None)
            s=A[n['infoset']]
            return sum(ev(a['child'],q*s[j],store) for j,a in enumerate(n['actions']))
        # process deepest infoset first
        for I in sorted(depth,key=lambda x:-depth[x]):
            pass
        # simple approach: iterate best-response fixed-point (perfect recall => converges in max-depth passes)
        for I,v in IS.items():
            if v['player']==pl: choice[I]=0
        for _ in range(max(depth.values())+2):
            store={}
            ev(g['root'],1.0,store)
            ch={I:max(range(len(d)),key=lambda j:d[j]) for I,d in store.items()}
            for I in IS:
                if IS[I]['player']==pl and I not in ch: ch[I]=choice[I]
            if ch==choice:break
            choice=ch
        res.append(ev(g['root'],1.0,None))  # value to pl... 
    # res[pl] = BR value for pl (sg applied twice => times sg); recompute properly
    return A,res
def main():
    T=int(sys.argv[1]);t0=time.time()
    for t in range(1,T+1):
        cfr(g['root'],1.0,1.0,t)
        if t%200==0: print(t,time.time()-t0,flush=True)
        if time.time()-t0>600:break
    A,res=expl()
    json.dump({"format":"bluff-strategy-v1","game":g['name'],"producer":"CFR+ linear avg, %d iters"%t,
      "probs":{k:{l:repr(p) for l,p in zip(IS[k]['labels'],A[k])} for k in IS}},open('strategy.json','w'),indent=1)
    # res[pl]: ev with sg multiplied once inside terminals then again here => BR value for pl in own payoff units is res*sg? compute directly
    print('BR0',res[0],'BR1',res[1],'expl',(res[0]+res[1])/2)
main()
