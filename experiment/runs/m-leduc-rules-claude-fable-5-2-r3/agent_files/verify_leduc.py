#!/usr/bin/env python3
"""Independent check of strategy.json for Leduc hold'em.

Shares no code with solve_leduc.py: the game is implemented again here as a plain state machine,
and values are computed deal by deal in exact rational arithmetic.

Usage: python3 verify_leduc.py [strategy path]
"""
import json
import sys
from fractions import Fraction

CARDS = ["Js", "Jh", "Qs", "Qh", "Ks", "Kh"]
RANKS = "JQK"
LETTER = {"fold": "f", "call": "c", "raise": "r"}


class State:
    """A point in the betting, for a fixed deal."""

    def __init__(self):
        self.pot = [1, 1]          # chips each player has put in
        self.round = 0             # 0 or 1
        self.hist = ["", ""]       # action letters per round
        self.raises = 0
        self.to_act = 0
        self.folded = None
        self.done = False

    def copy(self):
        s = State.__new__(State)
        s.pot = list(self.pot)
        s.round = self.round
        s.hist = list(self.hist)
        s.raises = self.raises
        s.to_act = self.to_act
        s.folded = self.folded
        s.done = self.done
        return s

    def legal(self):
        me, opp = self.to_act, 1 - self.to_act
        out = []
        if self.pot[opp] > self.pot[me]:
            out.append("fold")
        out.append("call")
        if self.raises < 2:
            out.append("raise")
        return out

    def step(self, action):
        s = self.copy()
        me, opp = s.to_act, 1 - s.to_act
        first = s.hist[s.round] == ""
        s.hist[s.round] += LETTER[action]
        if action == "fold":
            s.folded = me
            s.done = True
            return s
        if action == "raise":
            owed = s.pot[opp] - s.pot[me]
            s.pot[me] += owed + (2 if s.round == 0 else 4)
            s.raises += 1
            s.to_act = opp
            return s
        s.pot[me] = s.pot[opp]
        if first:
            s.to_act = opp
        elif s.round == 0:
            s.round, s.raises, s.to_act = 1, 0, 0
        else:
            s.done = True
        return s

    def name(self, card, public):
        if self.round == 0:
            return "%s:%s" % (card, self.hist[0])
        return "%s|%s:%s/%s" % (card, public, self.hist[0], self.hist[1])


def payoff0(state, c0, c1, public):
    if state.folded is not None:
        return -state.pot[0] if state.folded == 0 else state.pot[1]
    r0, r1, rp = RANKS.index(c0[0]), RANKS.index(c1[0]), RANKS.index(public[0])
    if r0 == rp:
        return state.pot[1]
    if r1 == rp:
        return -state.pot[0]
    if r0 > r1:
        return state.pot[1]
    if r1 > r0:
        return -state.pot[0]
    return 0


def publics(c0, c1):
    return [c for c in CARDS if c != c0 and c != c1]


def all_infosets():
    """Every information set name with its legal actions, found by walking the game."""
    found = {}

    def walk(state, cards, public):
        if state.done:
            return
        if state.round == 1 and public is None:
            for p in publics(*cards):
                walk(state, cards, p)
            return
        found[state.name(cards[state.to_act], public)] = state.legal()
        for a in state.legal():
            walk(state.step(a), cards, public)

    for c0 in CARDS:
        for c1 in CARDS:
            if c0 != c1:
                walk(State(), (c0, c1), None)
    return found


def load(path):
    with open(path) as f:
        doc = json.load(f)
    assert doc["format"] == "bluff-strategy-v1", doc["format"]
    assert doc["game"] == "leduc", doc["game"]
    probs = {k: {a: Fraction(p) for a, p in v.items()} for k, v in doc["probs"].items()}
    expected = all_infosets()
    missing = sorted(set(expected) - set(probs))
    extra = sorted(set(probs) - set(expected))
    assert not missing, "missing information sets: %s" % missing[:5]
    assert not extra, "unknown information sets: %s" % extra[:5]
    for name, acts in expected.items():
        assert sorted(probs[name]) == sorted(acts), (name, probs[name], acts)
        assert all(p >= 0 for p in probs[name].values()), name
        assert sum(probs[name].values()) == 1, (name, sum(probs[name].values()))
    return probs


def value_p0(probs):
    """Expected payoff to player 0 when both follow the strategy."""

    def walk(state, c0, c1, public):
        if state.done:
            return Fraction(payoff0(state, c0, c1, public))
        if state.round == 1 and public is None:
            ps = publics(c0, c1)
            return sum(walk(state, c0, c1, p) for p in ps) / len(ps)
        card = c0 if state.to_act == 0 else c1
        sigma = probs[state.name(card, public)]
        return sum(sigma[a] * walk(state.step(a), c0, c1, public)
                   for a in state.legal() if sigma[a] > 0)

    total = Fraction(0)
    for c0 in CARDS:
        for c1 in CARDS:
            if c0 != c1:
                total += walk(State(), c0, c1, None)
    return total / 30


def best_response_value(probs, hero):
    """Value to `hero` of the best response against the other player's strategy.

    `weights` maps each opponent card still possible to the probability that the deal gave it
    and the opponent played to this point (chance included). The hero sees only their own card
    and the public history, so one decision is made for the whole dictionary.
    """

    def walk(state, mine, public, weights):
        if not weights:
            return Fraction(0)
        if state.done:
            total = Fraction(0)
            for theirs, w in weights.items():
                c0, c1 = (mine, theirs) if hero == 0 else (theirs, mine)
                u0 = payoff0(state, c0, c1, public)
                total += w * (u0 if hero == 0 else -u0)
            return total
        if state.round == 1 and public is None:
            total = Fraction(0)
            for p in CARDS:
                if p == mine:
                    continue
                # Given both private cards, each of the 4 remaining cards has probability 1/4.
                nxt = {c: w / 4 for c, w in weights.items() if c != p}
                total += walk(state, mine, p, nxt)
            return total
        if state.to_act == hero:
            return max(walk(state.step(a), mine, public, weights) for a in state.legal())
        total = Fraction(0)
        for a in state.legal():
            nxt = {}
            for theirs, w in weights.items():
                p = probs[state.name(theirs, public)][a]
                if p > 0:
                    nxt[theirs] = w * p
            total += walk(state.step(a), mine, public, nxt)
        return total

    total = Fraction(0)
    for mine in CARDS:
        weights = {c: Fraction(1, 30) for c in CARDS if c != mine}
        total += walk(State(), mine, None, weights)
    return total


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "strategy.json"
    probs = load(path)
    print("information sets: %d, all present, all sum to exactly 1" % len(probs))
    value = value_p0(probs)
    br0 = best_response_value(probs, 0)
    br1 = best_response_value(probs, 1)
    expl = (br0 + br1) / 2
    print("value_p0            %+.12f" % float(value))
    print("best response p0    %+.12f" % float(br0))
    print("best response p1    %+.12f" % float(br1))
    print("exploitability      %.6e" % float(expl))
    assert br0 >= value and br1 >= -value and expl >= 0


if __name__ == "__main__":
    main()
