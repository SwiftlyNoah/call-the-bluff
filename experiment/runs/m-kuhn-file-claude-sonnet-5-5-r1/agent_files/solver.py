import json
from fractions import Fraction as F
g=json.load(open('game.json'))
N={n['id']:n for n in g['nodes']}
def num(s): return float(F(s))
IS={}
for n in N.values():
    if n['kind']=='decision':
        IS.setdefault(n['infoset'],{'p':n['player'],'a':[a['label'] for a in n['actions']]})
reg={k:[0.0]*len(v['a']) for k,v in IS.items()}
ssum={k:[0.0]*len(v['a']) for k,v in IS.items()}
def cur(k):
    r=[max(x,0) for x in reg[k]];s=sum(r)
    return [x/s for x in r] if s>0 else [1/len(r)]*len(r)
def cfr(i,r0,r1,r2):  # returns value for p0; r2 chance
    n=N[i]
    if n['kind']=='terminal': return num(n['payoff_p0'])
    if n['kind']=='chance':
        return sum(num(o['prob'])*cfr(o['child'],r0,r1,r2) for o in n['outcomes'])
    k=n['infoset'];p=n['player'];s=cur(k)
    vs=[cfr(a['child'],r0*(s[j] if p==0 else 1),r1*(s[j] if p==1 else 1),r2) for j,a in enumerate(n['actions'])]
    v=sum(x*y for x,y in zip(s,vs))
    mine,opp=(r0,r1) if p==0 else (r1,r0)
    for j in range(len(s)):
        d=(vs[j]-v) if p==0 else (v-vs[j])
        reg[k][j]=max(0,reg[k][j]+opp*d)
        ssum[k][j]+=mine*s[j]*(T+1)  # linear averaging
    return v
T=0
for T in range(20000): cfr(g['root'],1,1,1)
avg={}
for k in IS:
    s=sum(ssum[k]); n=len(ssum[k])
    avg[k]=[x/s for x in ssum[k]] if s>0 else [1/n]*n
# evaluation
def val(i,pi):
    n=N[i]
    if n['kind']=='terminal': return num(n['payoff_p0'])
    if n['kind']=='chance': return sum(num(o['prob'])*val(o['child'],pi) for o in n['outcomes'])
    s=pi[n['infoset']]
    return sum(s[j]*val(a['child'],pi) for j,a in enumerate(n['actions']))
def br(player):
    # collect (infoset -> action -> value weighted by opp+chance reach), then pick best
    ev={k:[0.0]*len(v['a']) for k,v in IS.items() if v['p']==player}
    def walk(i,reach,sel):
        n=N[i]
        if n['kind']=='terminal': return num(n['payoff_p0'])*(1 if player==0 else -1)*reach
        if n['kind']=='chance': return sum(walk(o['child'],reach*num(o['prob']),sel) for o in n['outcomes'])
        k=n['infoset']
        if n['player']==player:
            tot=0
            for j,a in enumerate(n['actions']):
                v=walk(a['child'],reach,sel); ev[k][j]+=v
            return 0
        s=avg[k]
        return sum(walk(a['child'],reach*s[j],sel) for j,a in enumerate(n['actions']))
    # need best-response values bottom-up: iterate to fixed point via pure strategy updates
    pure={k:0 for k in ev}
    def run(pure):
        for k in ev: ev[k]=[0.0]*len(ev[k])
        def w2(i,reach):
            n=N[i]
            if n['kind']=='terminal': return num(n['payoff_p0'])*(1 if player==0 else -1)*reach
            if n['kind']=='chance': return sum(w2(o['child'],reach*num(o['prob'])) for o in n['outcomes'])
            k=n['infoset']
            if n['player']==player:
                t=0
                for j,a in enumerate(n['actions']):
                    v=w2(a['child'],reach)
                    ev[k][j]+=v
                    if j==pure[k]: t=v
                return t
            return sum(w2(a['child'],reach*avg[k][j]) for j,a in enumerate(n['actions']))
        return w2(g['root'],1)
    # best response via backward induction: iterate policy improvement until stable
    while True:
        tot=run(pure)
        new={k:max(range(len(ev[k])),key=lambda j:(ev[k][j],-j)) for k in ev}
        # ev uses downstream pure choices, so improvement is valid
        if new==pure: return tot
        pure=new
v0=val(g['root'],avg)
e=(br(0)+br(1))/2
print('value',v0,'expl',e)
json.dump({'format':'bluff-strategy-v1','game':g['name'],'producer':'CFR+ linear averaging, 20000 iters',
 'probs':{k:{a:repr(avg[k][j]) for j,a in enumerate(IS[k]['a'])} for k in IS}},open('strategy.json','w'),indent=1)
