import json
from fractions import Fraction as F
g=json.load(open('game.json'))
N={n['id']:n for n in g['nodes']}
num=lambda s: float(F(s))
IS={}
for n in g['nodes']:
    if n['kind']=='decision':
        d=IS.setdefault(n['infoset'],dict(p=n['player'],a=[a['label'] for a in n['actions']]))
reg={k:[0.0]*len(v['a']) for k,v in IS.items()}
ssum={k:[0.0]*len(v['a']) for k,v in IS.items()}
def strat(k):
    r=reg[k];s=sum(r)
    return [x/s for x in r] if s>0 else [1/len(r)]*len(r)
def cfr(nid,c,r0,r1,upd,w):
    n=N[nid];k=n['kind']
    if k=='terminal': return num(n['payoff_p0'])
    if k=='chance':
        return sum(num(o['prob'])*cfr(o['child'],c*num(o['prob']),r0,r1,upd,w) for o in n['outcomes'])
    key=n['infoset'];s=strat(key);pl=n['player']
    vs=[cfr(a['child'],c,r0*s[i] if pl==0 else r0,r1*s[i] if pl==1 else r1,upd,w) for i,a in enumerate(n['actions'])]
    v=sum(x*y for x,y in zip(s,vs))
    if upd==pl:
        opp=r1 if pl==0 else r0; me=r0 if pl==0 else r1; sg=1 if pl==0 else -1
        for i in range(len(vs)):
            reg[key][i]=max(0.0,reg[key][i]+c*opp*sg*(vs[i]-v))
            ssum[key][i]+=w*c*me*s[i]
    return v
T=300000
for t in range(1,T+1):
    for u in (0,1): cfr(g['root'],1.0,1.0,1.0,u,t)
avg={}
for k in IS:
    s=sum(ssum[k]); avg[k]=[x/s for x in ssum[k]] if s>0 else [1/len(ssum[k])]*len(ssum[k])
# evaluation: value and exact best responses
def reach(nid,c,r0,r1,out):
    n=N[nid];k=n['kind']
    out[nid]=(c,r0,r1)
    if k=='chance':
        for o in n['outcomes']: reach(o['child'],c*num(o['prob']),r0,r1,out)
    elif k=='decision':
        for i,a in enumerate(n['actions']):
            p=avg[n['infoset']][i]
            reach(a['child'],c,r0*p if n['player']==0 else r0,r1*p if n['player']==1 else r1,out)
R={};reach(g['root'],1.0,1.0,1.0,R)
value=sum(R[i][0]*R[i][1]*R[i][2]*num(n['payoff_p0']) for i,n in N.items() if n['kind']=='terminal')
def br(pl):
    ch={}
    def val(nid):
        n=N[nid];k=n['kind']
        if k=='terminal': return num(n['payoff_p0'])*(1 if pl==0 else -1)*1.0
        if k=='chance': return sum(val(o['child']) for o in n['outcomes'])  # weights live in reach
        if n['player']==pl: return val_at(n,ch[n['infoset']])
        return sum(avg[n['infoset']][i]*0+val(a['child']) for i,a in enumerate(n['actions']))
    # use weighted values: value of node = sum over terminals of (c*opp reach)*payoff given BR
    def wv(nid):
        n=N[nid];k=n['kind']
        if k=='terminal':
            c,r0,r1=R[nid];opp=r1 if pl==0 else r0
            return c*opp*num(n['payoff_p0'])*(1 if pl==0 else -1)
        if k=='chance': return sum(wv(o['child']) for o in n['outcomes'])
        if n['player']==pl: return wv(n['actions'][ch[n['infoset']]]['child'])
        return sum(wv(a['child']) for a in n['actions'])
    def wv_(nid): return wv(nid)
    def depth(nid):
        n=N[nid]
        if n['kind']=='terminal': return 0
        ch_=[o['child'] for o in n['outcomes']] if n['kind']=='chance' else [a['child'] for a in n['actions']]
        return 1+max(depth(x) for x in ch_)
    mine=[k for k,v in IS.items() if v['p']==pl]
    nodes={k:[i for i,n in N.items() if n.get('infoset')==k] for k in mine}
    d={k:max(depth(i) for i in nodes[k]) for k in mine}
    for k in sorted(mine,key=lambda k:d[k]):
        best=max(range(len(IS[k]['a'])),key=lambda a:sum(wv(N[i]['actions'][a]['child']) for i in nodes[k]))
        ch[k]=best
    return wv(g['root'])
b0=br(0);b1=br(1)
expl=(b0+b1)/2
print(value,b0,b1,expl)
json.dump({"format":"bluff-strategy-v1","game":g['name'],"producer":"CFR+ linear averaging, %d iters"%T,
 "probs":{k:{a:repr(avg[k][i]) for i,a in enumerate(IS[k]['a'])} for k in IS}},open('strategy.json','w'),indent=1)
for k in sorted(avg): print(k,[round(x,4) for x in avg[k]])
