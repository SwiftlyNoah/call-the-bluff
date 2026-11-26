import json,sys,time
from fractions import Fraction
sys.setrecursionlimit(100000)
G=json.load(open('game.json'))
N={n['id']:n for n in G['nodes']}
F=lambda s: float(Fraction(s))
for n in N.values():
    if n['kind']=='chance':
        for o in n['outcomes']: o['p']=F(o['prob'])
    if n['kind']=='terminal': n['v']=F(n['payoff_p0'])
IS={}
for n in G['nodes']:
    if n['kind']=='decision':
        e=IS.setdefault(n['infoset'],(n['player'],[a['label'] for a in n['actions']],[]))
        e[2].append(n['id'])
reg={k:[0.0]*len(v[1]) for k,v in IS.items()}
ssum={k:[0.0]*len(v[1]) for k,v in IS.items()}
def cur(k):
    r=reg[k];s=sum(r);n=len(r)
    return [x/s for x in r] if s>0 else [1/n]*n
def avg(k):
    s=sum(ssum[k]);n=len(ssum[k])
    return [x/s for x in ssum[k]] if s>0 else [1/n]*n
def walk(i,p,t,rp,ro):
    # returns value for p; ro includes chance and opponent reach
    n=N[i];kd=n['kind']
    if kd=='terminal': return n['v'] if p==0 else -n['v']
    if kd=='chance':
        return sum(o['p']*walk(o['child'],p,t,rp,ro*o['p']) for o in n['outcomes'])
    k=n['infoset'];s=cur(k);acts=n['actions']
    if n['player']==p:
        vs=[walk(a['child'],p,t,rp*s[j],ro) for j,a in enumerate(acts)]
        v=sum(s[j]*vs[j] for j in range(len(s)))
        r=reg[k]
        for j in range(len(s)): r[j]=max(0.0,r[j]+ro*(vs[j]-v))
        sk=ssum[k]
        for j in range(len(s)): sk[j]+=t*rp*s[j]
        return v
    return sum(s[j]*walk(a['child'],p,t,rp,ro*s[j]) for j,a in enumerate(acts))

def ev(i,strat,memo):
    if i in memo: return memo[i]
    n=N[i];kd=n['kind']
    if kd=='terminal': v=n['v']
    elif kd=='chance': v=sum(o['p']*ev(o['child'],strat,memo) for o in n['outcomes'])
    else:
        s=strat[n['infoset']];v=sum(s[j]*ev(a['child'],strat,memo) for j,a in enumerate(n['actions']))
    memo[i]=v;return v
def br(p,strat):
    # opponent+chance reach per node
    reach={};stack=[(0,1.0)]
    while stack:
        i,r=stack.pop();reach[i]=r;n=N[i]
        if n['kind']=='chance':
            for o in n['outcomes']: stack.append((o['child'],r*o['p']))
        elif n['kind']=='decision':
            s=strat[n['infoset']]
            for j,a in enumerate(n['actions']):
                stack.append((a['child'],r*(s[j] if n['player']!=p else 1.0)))
    memo={};choice={}
    sg=1 if p==0 else -1
    def val(i):
        if i in memo: return memo[i]
        n=N[i];kd=n['kind']
        if kd=='terminal': v=sg*n['v']
        elif kd=='chance': v=sum(o['p']*val(o['child']) for o in n['outcomes'])
        elif n['player']!=p:
            s=strat[n['infoset']];v=sum(s[j]*val(a['child']) for j,a in enumerate(n['actions']))
        else:
            k=n['infoset']
            if k not in choice:
                m=len(n['actions']);tot=[0.0]*m
                for nid in IS[k][2]:
                    for j,a in enumerate(N[nid]['actions']): tot[j]+=reach[nid]*val(a['child'])
                choice[k]=max(range(m),key=lambda j:tot[j])
            v=val(n['actions'][choice[k]]['child'])
        memo[i]=v;return v
    return val(0)
def expl():
    st={k:avg(k) for k in IS}
    b0=br(0,st);b1=br(1,st)
    return (b0+b1)/2, ev(0,st,{})
T=int(sys.argv[1]) if len(sys.argv)>1 else 1000
t0=time.time()
for t in range(1,T+1):
    for p in (0,1): walk(0,p,t,1.0,1.0)
    if t%100==0 or t==T:
        e,v=expl();print(t,e,v,time.time()-t0,flush=True)
        if e<1e-4 or time.time()-t0>900: break
st={k:avg(k) for k in IS}
json.dump({"format":"bluff-strategy-v1","game":G['name'],"producer":"CFR+ (regret matching+, linear averaging)",
 "probs":{k:{l:repr(st[k][j]) for j,l in enumerate(IS[k][1])} for k in IS}},open('strategy.json','w'))
print("final",expl(),len(IS))
