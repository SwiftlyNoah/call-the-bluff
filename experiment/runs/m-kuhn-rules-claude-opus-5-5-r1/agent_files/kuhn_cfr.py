"""Kuhn poker solver using CFR+ (alternating updates, regret matching+, linear averaging)."""
import itertools, json

CARDS = "JQK"
RANK = {"J": 0, "Q": 1, "K": 2}
LETTER = {"check": "c", "bet": "b", "fold": "f", "call": "k"}
DEALS = list(itertools.permutations(CARDS, 2))  # 6 equally likely deals


def actions(h):
    return ["fold", "call"] if h and h[-1] == "b" else ["check", "bet"]


def terminal_payoff_p0(h, c0, c1):
    """Net payoff to player 0 at terminal history h, or None if not terminal."""
    win = 1 if RANK[c0] > RANK[c1] else -1
    if h == "cc":
        return win
    if h in ("bk", "cbk"):
        return 2 * win
    if h == "bf":
        return 1  # P1 folded
    if h == "cbf":
        return -1  # P0 folded
    return None


def infosets():
    return [f"{c}:{h}" for c in CARDS for h in ["", "c", "b", "cb"]]


def player(h):
    return len(h) % 2


def label(h):  # infoset history string uses only c/b
    return h


class CFRPlus:
    def __init__(self):
        self.regret = {I: {a: 0.0 for a in actions(I.split(":")[1])} for I in infosets()}
        self.avg = {I: {a: 0.0 for a in actions(I.split(":")[1])} for I in infosets()}

    def current(self, I):
        r = self.regret[I]
        pos = {a: max(v, 0.0) for a, v in r.items()}
        s = sum(pos.values())
        return {a: (v / s if s > 0 else 1 / len(r)) for a, v in pos.items()}

    def cfr(self, h, cards, traverser, pi_opp, pi_self, t):
        u = terminal_payoff_p0(h, *cards)
        if u is not None:
            return u if traverser == 0 else -u
        p = player(h)
        I = f"{cards[p]}:{h}"
        sigma = self.sigma[I]
        acts = actions(h)
        if p == traverser:
            vals = {}
            for a in acts:
                vals[a] = self.cfr(h + LETTER[a], cards, traverser, pi_opp, pi_self * sigma[a], t)
            v = sum(sigma[a] * vals[a] for a in acts)
            for a in acts:
                self.delta[I][a] += pi_opp * (vals[a] - v)
            for a in acts:  # linear averaging weighted by own reach
                self.avg[I][a] += t * pi_self * sigma[a]
            return v
        v = 0.0
        for a in acts:
            v += sigma[a] * self.cfr(h + LETTER[a], cards, traverser, pi_opp * sigma[a], pi_self, t)
        return v

    def train(self, iters):
        for t in range(1, iters + 1):
            for trav in (0, 1):
                # freeze current strategies for the whole iteration, then apply regret+ update
                self.sigma = {I: self.current(I) for I in self.regret}
                self.delta = {I: {a: 0.0 for a in r} for I, r in self.regret.items()}
                for d in DEALS:
                    self.cfr("", d, trav, 1 / 6, 1.0, t)
                for I, r in self.regret.items():
                    for a in r:
                        r[a] = max(r[a] + self.delta[I][a], 0.0)

    def average(self):
        out = {}
        for I, s in self.avg.items():
            tot = sum(s.values())
            out[I] = {a: (v / tot if tot > 0 else 1 / len(s)) for a, v in s.items()}
        return out


# ---------- evaluation ----------
def expected_value_p0(strat, h="", cards=None):
    if cards is None:
        return sum(expected_value_p0(strat, "", d) for d in DEALS) / 6
    u = terminal_payoff_p0(h, *cards)
    if u is not None:
        return u
    I = f"{cards[player(h)]}:{h}"
    return sum(p * expected_value_p0(strat, h + LETTER[a], cards) for a, p in strat[I].items())


def best_response_value(strat, br):
    """Value to player br when best-responding to strat for the other player (exact, infoset-level)."""
    def val(h, mycard):
        # returns value to br summed over opponent cards weighted by chance*opp reach
        # opp_reach: dict card -> prob
        pass

    def rec(h, mycard, opp_reach):
        total = 0.0
        if terminal_payoff_p0(h, "J", "Q") is not None:
            for oc, r in opp_reach.items():
                c = (mycard, oc) if br == 0 else (oc, mycard)
                u = terminal_payoff_p0(h, *c)
                total += r * (u if br == 0 else -u)
            return total
        p = player(h)
        acts = actions(h)
        if p == br:
            return max(rec(h + LETTER[a], mycard, opp_reach) for a in acts)
        for a in acts:
            nr = {oc: r * strat[f"{oc}:{h}"][a] for oc, r in opp_reach.items()}
            total += rec(h + LETTER[a], mycard, nr)
        return total

    v = 0.0
    for mc in CARDS:
        opp = {oc: 1 / 6 for oc in CARDS if oc != mc}
        v += rec("", mc, opp)
    return v


def exploitability(strat):
    return (best_response_value(strat, 0) + best_response_value(strat, 1)) / 2


if __name__ == "__main__":
    solver = CFRPlus()
    solver.train(20000)
    strat = solver.average()
    ex = exploitability(strat)
    v0 = expected_value_p0(strat)
    print(f"exploitability={ex:.3e} value_p0={v0:.6f} (theory -1/18={-1/18:.6f})")
    for I in infosets():
        print(I, {a: round(p, 4) for a, p in strat[I].items()})
    with open("strategy.json", "w") as f:
        json.dump({
            "format": "bluff-strategy-v1",
            "game": "kuhn",
            "producer": "CFR+ (alternating updates, linear averaging), 20000 iterations, kuhn_cfr.py",
            "probs": {I: {a: repr(p) for a, p in s.items()} for I, s in strat.items()},
        }, f, indent=2)
    print("wrote strategy.json")
