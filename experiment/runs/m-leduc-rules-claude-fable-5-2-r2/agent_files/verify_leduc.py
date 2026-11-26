#!/usr/bin/env python3
"""Independent check of strategy.json for Leduc hold'em.

Shares no code with solve_leduc.py. It walks the full game tree deal by deal in exact
rational arithmetic, checks that the file lists exactly the game's information sets and
legal actions with probabilities summing to 1, and computes the value of the profile and
each player's best-response value.

Usage: python3 verify_leduc.py [strategy.json]
"""
import itertools
import json
import sys
from fractions import Fraction

DECK = ["Js", "Jh", "Qs", "Qh", "Ks", "Kh"]
ORDER = "JQK"
BET = {1: 2, 2: 4}


def legal(history):
    """Legal actions and the player to act, or a terminal marker, for a betting history.

    `history` is a tuple (round-1 letters, round-2 letters or None). Returns
    ("terminal", payoff-kind, data) or ("act", player, actions, chips).
    """
    first, second = history
    chips = [1, 1]
    result = None
    for rnd, letters in ((1, first), (2, second)):
        if letters is None:
            break
        raises = 0
        mover = 0
        closed = False
        for n, ch in enumerate(letters):
            assert not closed, history
            other = 1 - mover
            if ch == "f":
                assert chips[mover] < chips[other], history
                return ("fold", mover, chips)
            if ch == "c":
                chips[mover] = chips[other]
                if n > 0:
                    closed = True
            elif ch == "r":
                assert raises < 2, history
                chips[mover] = chips[other] + BET[rnd]
                raises += 1
            else:
                raise ValueError(history)
            mover = other
        if not closed:
            actions = []
            if chips[mover] < chips[1 - mover]:
                actions.append("fold")
            actions.append("call")
            if raises < 2:
                actions.append("raise")
            return ("act", mover, actions, rnd)
        result = rnd
    if result == 1:
        return ("deal", None, chips)
    return ("showdown", None, chips)


def winner(c0, c1, pub):
    """+1 if player 0 wins the showdown, -1 if player 1 wins, 0 for a split."""
    if c0[0] == pub[0]:
        return 1
    if c1[0] == pub[0]:
        return -1
    a, b = ORDER.index(c0[0]), ORDER.index(c1[0])
    return (a > b) - (a < b)


def infoset_name(card, pub, history):
    first, second = history
    if second is None:
        return f"{card}:{first}"
    return f"{card}|{pub}:{first}/{second}"


def step(history, action):
    letter = action[0]
    first, second = history
    if second is None:
        return (first + letter, None)
    return (first, second + letter)


class Game:
    def __init__(self, probs):
        self.probs = probs
        self.seen = {}

    def strategy(self, name, actions):
        self.seen[name] = actions
        entry = self.probs[name]
        return [entry[a] for a in actions]

    def walk(self, deal, history, visit):
        """Generic recursion. `visit` gets (kind, ...) and combines child values."""
        c0, c1, pub = deal
        state = legal(history)
        if state[0] == "fold":
            _, folder, chips = state
            return visit.leaf(Fraction(-chips[0] if folder == 0 else chips[1]))
        if state[0] == "showdown":
            return visit.leaf(Fraction(state[2][0] * winner(c0, c1, pub)))
        if state[0] == "deal":
            return self.walk(deal, (history[0], ""), visit)
        _, mover, actions, rnd = state
        card = c0 if mover == 0 else c1
        name = infoset_name(card, pub if rnd == 2 else None, history)
        return visit.decision(self, deal, history, mover, name, actions)


class Evaluate:
    """Expected payoff to player 0 when both follow the file."""

    def leaf(self, payoff0):
        return payoff0

    def decision(self, game, deal, history, mover, name, actions):
        ps = game.strategy(name, actions)
        return sum(p * game.walk(deal, step(history, a), self)
                   for p, a in zip(ps, actions) if p)


class Collect:
    """First pass of a best response: opponent-reach weight of every node of `hero`."""

    def __init__(self, hero):
        self.hero = hero
        self.members = {}
        self.weight = Fraction(1)

    def leaf(self, payoff0):
        return None

    def decision(self, game, deal, history, mover, name, actions):
        if mover == self.hero:
            self.members.setdefault(name, []).append((deal, history, self.weight))
            for a in actions:
                game.walk(deal, step(history, a), self)
            return None
        ps = game.strategy(name, actions)
        saved = self.weight
        for p, a in zip(ps, actions):
            if p:
                self.weight = saved * p
                game.walk(deal, step(history, a), self)
        self.weight = saved
        return None


class Respond:
    """Second pass: the hero picks, at each information set, the action that is best
    against the opponent's strategy summed over every node of that information set."""

    def __init__(self, hero, members, actions_of):
        self.hero = hero
        self.members = members
        self.actions_of = actions_of
        self.choice = {}
        self.memo = {}

    def leaf(self, payoff0):
        return payoff0 if self.hero == 0 else -payoff0

    def pick(self, game, name):
        if name not in self.choice:
            actions = self.actions_of[name]
            best = None
            for a in actions:
                total = sum(w * self.value(game, deal, step(history, a))
                            for deal, history, w in self.members[name])
                if best is None or total > best[0]:
                    best = (total, a)
            self.choice[name] = best[1]
        return self.choice[name]

    def value(self, game, deal, history):
        key = (deal, history)
        if key not in self.memo:
            self.memo[key] = game.walk(deal, history, self)
        return self.memo[key]

    def decision(self, game, deal, history, mover, name, actions):
        if mover == self.hero:
            return self.value(game, deal, step(history, self.pick(game, name)))
        ps = game.strategy(name, actions)
        return sum(p * self.value(game, deal, step(history, a))
                   for p, a in zip(ps, actions) if p)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "strategy.json"
    with open(path) as f:
        doc = json.load(f)
    assert doc["format"] == "bluff-strategy-v1", doc["format"]
    assert doc["game"] == "leduc", doc["game"]
    probs = {name: {a: Fraction(s) for a, s in entry.items()}
             for name, entry in doc["probs"].items()}
    for name, entry in doc["probs"].items():
        assert all(isinstance(s, str) for s in entry.values()), name
    for name, entry in probs.items():
        assert all(p >= 0 for p in entry.values()), name
        assert sum(entry.values()) == 1, (name, sum(entry.values()))

    deals = list(itertools.permutations(DECK, 3))
    assert len(deals) == 120
    chance = Fraction(1, len(deals))
    game = Game(probs)

    # Enumerate every information set (walking every action, whatever its probability).
    full = Collect(hero=0)
    full_other = Collect(hero=1)
    actions_of = {}

    class Enumerate:
        def leaf(self, payoff0):
            return None

        def decision(self, g, deal, history, mover, name, actions):
            prev = actions_of.setdefault(name, actions)
            assert prev == actions, name
            for a in actions:
                g.walk(deal, step(history, a), self)

    for deal in deals:
        game.walk(deal, ("", None), Enumerate())
    missing = sorted(set(actions_of) - set(probs))
    extra = sorted(set(probs) - set(actions_of))
    assert not missing, f"missing information sets: {missing[:5]}"
    assert not extra, f"unknown information sets: {extra[:5]}"
    for name, actions in actions_of.items():
        assert set(probs[name]) == set(actions), (name, probs[name], actions)
    print(f"information sets: {len(actions_of)} (all present, legal actions only, sums = 1)")

    value0 = sum(chance * game.walk(deal, ("", None), Evaluate()) for deal in deals)
    print(f"value to player 0 under the profile: {float(value0):+.12f}")

    br = []
    for hero, collector in ((0, full), (1, full_other)):
        for deal in deals:
            collector.weight = chance
            game.walk(deal, ("", None), collector)
        responder = Respond(hero, collector.members, actions_of)
        # Information sets the opponent's strategy never reaches have no members; any
        # action is a best response there.
        for name in actions_of:
            collector.members.setdefault(name, [])
        total = sum(chance * responder.value(game, deal, ("", None)) for deal in deals)
        br.append(total)
        print(f"best-response value of player {hero}: {float(total):+.12f}")
    expl = (br[0] + br[1]) / 2
    print(f"exploitability (br0 + br1) / 2: {float(expl):.6e}")
    print(json.dumps({"exploitability": float(expl), "value_p0": float(value0),
                      "br0": float(br[0]), "br1": float(br[1]),
                      "infosets": len(actions_of)}))


if __name__ == "__main__":
    main()
