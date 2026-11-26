import json,sys
from fractions import Fraction
sys.setrecursionlimit(10000)
g=json.load(open('game.json'))
N={n['id']:n for n in g['nodes']}
F=lambda s:float(Fraction(s))
IS={}  # infoset -> (player, labels)
for n in N.values():
    if n['kind']=='decision':
        IS.setdefault(n['infoset'],(n['player'],[a['label'] for a in n['actions']]))
reg={i:[0.0]*len(v[1]) for i,v in IS.items()}
ssum={i:[0.0]*len(v[1]) for i,v in IS.items()}
def strat(i):
    r=[max(x,0) for x in reg[i]];s=sum(r);k=len(r)
    return [x/s for x in r] if s>0 else [1/k]*k
def cfr(id,p0,p1,w):
    n=N[id];k=n['kind']
    if k=='terminal': return F(n['payoff_p0'])
    if k=='chance':
        return sum(F(o['prob'])*cfr(o['child'],p0,p1,w) for o in n['outcomes'])
    i=n['infoset'];pl=n['player'];s=strat(i)
    vs=[];v=0
    for a,pa in zip(n['actions'],s):
        x=cfr(a['child'],p0*pa if pl==0 else p0,p1*pa if pl==1 else p1,w);vs.append(x);v+=pa*x
    opp=p1 if pl==0 else p0;me=p0 if pl==0 else p1
    sg=1 if pl==0 else -1
    for j,x in enumerate(vs):
        reg[i][j]=max(0,reg[i][j]+opp*sg*(x-v))
        ssum[i][j]+=w*me*s[j]
    return v
T=int(sys.argv[1]) if len(sys.argv)>1 else 2000
for t in range(1,T+1): cfr(g['root'],1,1,t)
avg={}
for i in IS:
    s=sum(ssum[i]);k=len(ssum[i])
    avg[i]=[x/s for x in ssum[i]] if s>0 else [1/k]*k
def value(id,pr):
    n=N[id];k=n['kind']
    if k=='terminal':return F(n['payoff_p0'])
    if k=='chance':return sum(F(o['prob'])*value(o['child'],pr) for o in n['outcomes'])
    return sum(p*value(a['child'],pr) for a,p in zip(n['actions'],avg[n['infoset']]))
def br(P):
    sg=1 if P==0 else -1
    nodes={}
    memoW={};best={}
    # collect nodes per infoset with opp reach
    reach={}
    def walk(id,r):
        n=N[id];k=n['kind']
        if k=='terminal':return
        if k=='chance':
            for o in n['outcomes']:walk(o['child'],r*F(o['prob']))
            return
        if n['player']==P: nodes.setdefault(n['infoset'],[]).append(id)
        for a,p in zip(n['actions'],avg[n['infoset']]):
            walk(a['child'],r*(1 if n['player']==P else p))
        reach[id]=r
    walk(g['root'],1.0)
    def W(id):
        if id in memoW:return memoW[id]
        n=N[id];k=n['kind']
        if k=='terminal':
            r=sg*F(n['payoff_p0'])*TR[id]
        elif k=='chance':r=sum(W(o['child']) for o in n['outcomes'])
        elif n['player']==P:
            r=Q(n['infoset'])[1][n['actions'] and best[n['infoset']] if False else 0] if False else Wbest(n)
        else:r=sum(W(a['child']) for a in n['actions'])
        memoW[id]=r;return r
    def Wbest(n):
        i=n['infoset']
        if i not in best:
            q=[sum(W(N[h]['actions'][j]['child']) for h in nodes[i]) for j in range(len(IS[i][1]))]
            best[i]=max(range(len(q)),key=lambda j:q[j])
        return W(n['actions'][best[i]]['child'])
    # terminal reach: opp+chance prob
    TR={}
    def walk2(id,r):
        n=N[id];k=n['kind']
        if k=='terminal':TR[id]=r;return
        if k=='chance':
            for o in n['outcomes']:walk2(o['child'],r*F(o['prob']))
            return
        for a,p in zip(n['actions'],avg[n['infoset']]):
            walk2(a['child'],r*(1 if n['player']==P else p))
    walk2(g['root'],1.0)
    return W(g['root'])
b0=br(0);b1=br(1)
expl=(b0+b1)/2;v=value(g['root'],0)
json.dump({"format":"bluff-strategy-v1","game":g['name'],"producer":"CFR+ linear averaging, %d iters"%T,
 "probs":{i:{l:repr(p) for l,p in zip(IS[i][1],avg[i])} for i in IS}},open('strategy.json','w'),indent=1)
print(len(IS),b0,b1,expl,v)
