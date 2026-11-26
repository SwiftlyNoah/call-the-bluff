import json,sys
from fractions import Fraction
sys.setrecursionlimit(100000)
g=json.load(open('game.json'))
N={n['id']:n for n in g['nodes']}
fr=lambda s: float(Fraction(s))
IS={}  # infoset -> (player, labels)
for n in g['nodes']:
    if n['kind']=='decision':
        IS.setdefault(n['infoset'],(n['player'],[a['label'] for a in n['actions']]))
reg={k:[0.0]*len(v[1]) for k,v in IS.items()}
ssum={k:[0.0]*len(v[1]) for k,v in IS.items()}
def strat(k):
    r=[max(x,0) for x in reg[k]];s=sum(r)
    return [x/s for x in r] if s>0 else [1/len(r)]*len(r)
def cfr(i,p,r0,r1,t):
    n=N[i];k=n['kind']
    if k=='terminal': return fr(n['payoff_p0'])
    if k=='chance':
        return sum(fr(o['prob'])*cfr(o['child'],p,r0,r1,t) for o in n['outcomes'])
    h=n['infoset'];pl=n['player'];s=strat(h);ch=[a['child'] for a in n['actions']]
    if pl!=p:
        v=0
        for a,c in enumerate(ch):
            v+=s[a]*(cfr(c,p,r0,r1*s[a],t) if pl==1 else cfr(c,p,r0*s[a],r1,t))
        if pl==1: pass
        # accumulate average for opponent-of-updater
        return v
    vs=[cfr(c,p,r0*s[a] if pl==0 else r0, r1*s[a] if pl==1 else r1,t) for a,c in enumerate(ch)]
    v=sum(s[a]*vs[a] for a in range(len(ch)))
    opp=r1 if pl==0 else r0; sg=1 if pl==0 else -1
    for a in range(len(ch)):
        reg[h][a]=max(0.0,reg[h][a]+sg*opp*(vs[a]-v))
    own=r0 if pl==0 else r1
    for a in range(len(ch)): ssum[h][a]+=t*own*s[a]
    return v
def avg(k):
    s=sum(ssum[k]);return [x/s for x in ssum[k]] if s>0 else [1/len(ssum[k])]*len(ssum[k])
def evalv(i,S):
    n=N[i];k=n['kind']
    if k=='terminal': return fr(n['payoff_p0'])
    if k=='chance': return sum(fr(o['prob'])*evalv(o['child'],S) for o in n['outcomes'])
    s=S[n['infoset']];return sum(s[a]*evalv(x['child'],S) for a,x in enumerate(n['actions']))
def br(p,S):
    BR={k:0 for k,v in IS.items() if v[0]==p}
    sg=1 if p==0 else -1
    for _ in range(60):
        cf={k:[0.0]*len(IS[k][1]) for k in BR}
        def go(i,reach):  # reach: opp*chance prob; returns value to p
            n=N[i];k=n['kind']
            if k=='terminal': return sg*fr(n['payoff_p0'])
            if k=='chance': return sum(fr(o['prob'])*go(o['child'],reach) for o in n['outcomes'])
            h=n['infoset']
            if n['player']!=p:
                s=S[h];return sum(s[a]*go(x['child'],reach*s[a]) for a,x in enumerate(n['actions']))
            vs=[go(x['child'],reach) for x in n['actions']]
            for a,v in enumerate(vs): cf[h][a]+=reach*v
            return vs[BR[h]]
        val=go(g['root'],1.0)
        new={k:max(range(len(cf[k])),key=lambda a:cf[k][a]) for k in BR}
        if new==BR: return val
        BR=new
    return val
def expl(S): return (br(0,S)+br(1,S))/2
T=int(sys.argv[1]) if len(sys.argv)>1 else 2000
for t in range(1,T+1):
    for p in (0,1): cfr(g['root'],p,1.0,1.0,t)
    if t%250==0 or t==T:
        S={k:avg(k) for k in IS}
        print(t,expl(S),evalv(g['root'],S),flush=True)
S={k:avg(k) for k in IS}
json.dump({"format":"bluff-strategy-v1","game":g['name'],"producer":"CFR+ linear avg, alternating","probs":{k:{l:repr(S[k][a]) for a,l in enumerate(IS[k][1])} for k in IS}},open('strategy.json','w'))
