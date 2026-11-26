"""Tiny games written for attempt-b's tests (independent of games/). Writes tests/games/*.json.

uv run dafny/attempt-b/tests/gen_games.py
"""
import json
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "games")


class B:
    def __init__(self):
        self.nodes = []

    def add(self, nd):
        nd["id"] = len(self.nodes)
        self.nodes.append(nd)
        return nd["id"]

    def term(self, u):
        return self.add({"kind": "terminal", "payoff_p0": str(u)})

    def dec(self, player, infoset, actions):  # actions: list of (label, builder)
        i = self.add({"kind": "decision", "player": player, "infoset": infoset, "actions": []})
        self.nodes[i]["actions"] = [{"label": l, "child": f()} for l, f in actions]
        return i

    def chance(self, outcomes):  # (label, prob, builder)
        i = self.add({"kind": "chance", "outcomes": []})
        self.nodes[i]["outcomes"] = [{"label": l, "prob": p, "child": f()} for l, p, f in outcomes]
        return i


def game(name, desc, build):
    b = B()
    root = build(b)
    return {"format": "bluff-game-v1", "name": name, "description": desc, "root": root,
            "nodes": sorted(b.nodes, key=lambda n: n["id"])}


def pennies(b):
    return b.dec(0, "p0", [(a, (lambda a=a: b.dec(1, "p1", [(c, (lambda a=a, c=c: b.term(1 if a == c else -1)))
                                                           for c in "HT"]))) for a in "HT"])


def mini_kuhn(b):
    # 2 cards (L, H), ante 1, p0 checks or bets 1; after a check p1 checks or bets; calls/folds.
    def showdown(c0, c1, pot):
        return pot if c0 > c1 else -pot

    cards = {"L": 0, "M": 1, "H": 2}
    deals = [(x, y) for x in "LMH" for y in "LMH" if x != y]

    def mk(x, y):
        # infosets use card names, showdown compares ranks
        def f():
            c0, c1 = cards[x], cards[y]
            n = b.dec(0, f"{x}:", [])
            def p1c():
                return b.dec(1, f"{y}:c", [
                    ("check", lambda: b.term(showdown(c0, c1, 1))),
                    ("bet", lambda: b.dec(0, f"{x}:cb", [
                        ("fold", lambda: b.term(-1)),
                        ("call", lambda: b.term(showdown(c0, c1, 2)))]))])
            def p1b():
                return b.dec(1, f"{y}:b", [
                    ("fold", lambda: b.term(1)),
                    ("call", lambda: b.term(showdown(c0, c1, 2)))])
            b.nodes[n]["actions"] = [{"label": "check", "child": p1c()}, {"label": "bet", "child": p1b()}]
            return n
        return f
    return b.chance([(x + y, "1/6", mk(x, y)) for x, y in deals])


def chance_twice(b):
    # chance, p0 moves, chance again, p1 moves without seeing the second chance draw.
    def after(c, a):
        return lambda: b.chance([
            ("x", "1/3", lambda: b.dec(1, f"p1:{a}", [("l", lambda: b.term(2 if c == "u" else -1)),
                                                      ("r", lambda: b.term(-2 if a == "A" else 1))])),
            ("y", "2/3", lambda: b.dec(1, f"p1:{a}", [("l", lambda: b.term("1/2")),
                                                      ("r", lambda: b.term(0 if c == "u" else 3))]))])
    return b.chance([(c, p, (lambda c=c: b.dec(0, f"p0:{c}", [(a, after(c, a)) for a in "AB"])))
                     for c, p in (("u", "1/4"), ("d", "3/4"))])


def forgetful(b):
    # NOT perfect recall. A fair coin: on heads p0 first picks at "A" and then reaches "later";
    # on tails p0 reaches "later" directly. Heads pays 1 if A=L and later=L; tails pays 1 if
    # later=R. Every pure strategy gets at most 1/2, but deciding "later" separately in the two
    # branches would claim 1, so the core must refuse (best response not attained).
    def later(win):
        return b.dec(0, "p0:later", [(s, lambda s=s: b.term(1 if s == win else 0)) for s in "LR"])
    return b.chance([
        ("h", "1/2", lambda: b.dec(0, "p0:A", [("L", lambda: later("L")), ("R", lambda: b.term(0))])),
        ("t", "1/2", lambda: later("R"))])


GAMES = {
    "pennies_b": ("matching pennies, written for attempt-b", pennies),
    "minikuhn_b": ("3-card Kuhn-like poker, written for attempt-b", mini_kuhn),
    "chance_twice_b": ("chance at two depths, opponent blind to the second draw", chance_twice),
    "forgetful_b": ("imperfect recall: player 0 forgets its own first move", forgetful),
}

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for name, (desc, f) in GAMES.items():
        with open(os.path.join(OUT, name + ".game.json"), "w") as fh:
            json.dump(game(name, desc, f), fh, indent=1)
        print("wrote", name)
