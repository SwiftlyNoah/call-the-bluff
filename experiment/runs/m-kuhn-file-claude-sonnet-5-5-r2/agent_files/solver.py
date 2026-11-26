import json
from fractions import Fraction as F
g=json.load(open('game.json'))
N={n['id']:n for n in g['nodes']}
def num(s): return float(F(s))
IS={}
for n in N.values():
    if n['kind']=='decision': IS.setdefault(n['infoset'],(n['player'],[a['label'] for a in n['actions']]))
reg={k:[0.0]*len(v[1]) for k,v in IS.items()}
ssum={k:[0.0]*len(v[1]) for k,v in IS.items()}
def strat(k):
    r=[max(x,0) for x in reg[k]];s=sum(r);m=len(r)
    return [x/s for x in r] if s>0 else [1/m]*m
def cfr(i,p,r0,r1,w,sf):
    n=N[i];k=n['kind']
    if k=='terminal': return num(n['payoff_p0'])
    if k=='chance': return sum(num(o['prob'])*cfr(o['child'],p,r0,r1,w,sf) for o in n['outcomes'])
    key=n['infoset'];pl=n['player'];s=sf(key)
    vs=[]
    for a,pa in zip(n['actions'],s):
        vs.append(cfr(a['child'],p,r0*pa if pl==0 else r0,r1*pa if pl==1 else r1,w,sf))
    v=sum(a*b for a,b in zip(s,vs))
    if w is not None:
        opp=r1 if pl==0 else r0; me=r0 if pl==0 else r1
        sg=1 if pl==0 else -1
        for j in range(len(vs)):
            reg[key][j]=max(0,reg[key][j]+opp*sg*(vs[j]-v))
            ssum[key][j]+=w*me*s[j]
    return v
T=20000
for t in range(1,T+1): cfr(g['root'],0,1,1,t,strat)
avg={}
for k in IS:
    s=sum(ssum[k]); m=len(ssum[k])
    avg[k]=[x/s for x in ssum[k]] if s>0 else [1/m]*m
# exact exploitability: BR via infoset values
def br(player):
    # collect per-infoset action values weighted by opp+chance reach
    cv={k:[0.0]*len(v[1]) for k,v in IS.items() if v[0]==player}
    def walk(i,reach,pick):
        n=N[i];k=n['kind']
        if k=='terminal': return num(n['payoff_p0'])*(1 if player==0 else -1)
        if k=='chance': return sum(num(o['prob'])*walk(o['child'],reach*num(o['prob']),pick) for o in n['outcomes'])
        key=n['infoset']
        if n['player']==player:
            tot=0
            for j,a in enumerate(n['actions']):
                v=walk(a['child'],reach,pick)
                if pick is None: cv[key][j]+=reach*v
                elif pick[key]==j: tot=v
            return tot
        return sum(pa*walk(a['child'],reach*pa,pick) for a,pa in zip(n['actions'],avg[key]))
    # walk reach should include only opp/chance; chance prob multiplied in via reach arg above
    # iterate: compute pick by reverse-depth is complex; do fixed-point iteration
    pick={k:0 for k in cv}
    for _ in range(len(cv)+2):
        for k in cv: cv[k]=[0.0]*len(cv[k])
        # values depend on later picks; compute with current pick, collecting cv
        def w2(i,reach):
            n=N[i];kd=n['kind']
            if kd=='terminal': return num(n['payoff_p0'])*(1 if player==0 else -1)
            if kd=='chance': return sum(num(o['prob'])*w2(o['child'],reach*num(o['prob'])) for o in n['outcomes'])
            key=n['infoset']
            if n['player']==player:
                vs=[w2(a['child'],reach) for a in n['actions']]
                for j,v in enumerate(vs): cv[key][j]+=reach*v
                return vs[pick[key]]
            return sum(pa*w2(a['child'],reach*pa) for a,pa in zip(n['actions'],avg[key]))
        val=w2(g['root'],1.0)
        new={k:max(range(len(c)),key=lambda j:c[j]) for k,c in cv.items()}
        if new==pick: break
        pick=new
    return val
def value():
    def w(i):
        n=N[i];k=n['kind']
        if k=='terminal': return num(n['payoff_p0'])
        if k=='chance': return sum(num(o['prob'])*w(o['child']) for o in n['outcomes'])
        return sum(p*w(a['child']) for a,p in zip(n['actions'],avg[n['infoset']]))
    return w(g['root'])
b0,b1=br(0),br(1)
print('br',b0,b1,'expl',(b0+b1)/2,'value',value())
out={"format":"bluff-strategy-v1","game":g['name'],"producer":"CFR+ linear averaging, %d iters"%T,
 "probs":{k:{l:repr(p) for l,p in zip(IS[k][1],avg[k])} for k in IS}}
json.dump(out,open('strategy.json','w'),indent=1)
for k in sorted(avg): print(k,[round(x,3) for x in avg[k]])
