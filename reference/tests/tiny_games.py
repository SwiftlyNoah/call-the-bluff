"""Hand-made tiny games (as bluff-game-v1 dicts) and a random perfect-recall game generator."""

from __future__ import annotations

import random
from fractions import Fraction


class _Builder:
    def __init__(self):
        self.nodes: list[dict] = []

    def _new(self, node: dict) -> int:
        node["id"] = len(self.nodes)
        self.nodes.append(node)
        return node["id"]

    def terminal(self, payoff) -> int:
        return self._new({"kind": "terminal", "payoff_p0": str(payoff)})

    def chance(self, outcomes: list[tuple[str, str, int]]) -> int:
        return self._new({"kind": "chance", "outcomes": [
            {"label": lab, "prob": prob, "child": c} for lab, prob, c in outcomes]})

    def decision(self, player: int, infoset: str, actions: list[tuple[str, int]]) -> int:
        return self._new({"kind": "decision", "player": player, "infoset": infoset,
                          "actions": [{"label": lab, "child": c} for lab, c in actions]})

    def game(self, name: str, root: int, **extra) -> dict:
        return {"format": "bluff-game-v1", "name": name, "root": root,
                "nodes": self.nodes, **extra}


def matching_pennies() -> dict:
    """p0 picks H/T, p1 picks H/T without seeing it; p0 wins 1 on a match, loses 1 otherwise."""
    b = _Builder()
    kids = {}
    for a0 in "HT":
        p1 = b.decision(1, "p1", [(a1, b.terminal(1 if a0 == a1 else -1)) for a1 in "HT"])
        kids[a0] = p1
    root = b.decision(0, "p0", [(a, kids[a]) for a in "HT"])
    return b.game("matching-pennies", root, known_value_p0="0",
                  known_value_source="symmetry")


def bet_or_fold() -> dict:
    """Chance deals p0 a High card (1/3) or Low card (2/3). p0 sees it and bets or folds (-1).
    p1 sees only the bet and calls or folds (+1 to p0). On a call p0 wins 2 with High, loses 2
    with Low. Two p0 infosets ("H", "L") and one p1 infoset ("bet")."""
    b = _Builder()
    deal = []
    for card, prob in (("H", "1/3"), ("L", "2/3")):
        call = b.terminal(2 if card == "H" else -2)
        p1 = b.decision(1, "bet", [("call", call), ("fold", b.terminal(1))])
        p0 = b.decision(0, card, [("bet", p1), ("fold", b.terminal(-1))])
        deal.append((card, prob, p0))
    root = b.chance(deal)
    return b.game("bet-or-fold", root)


def two_stage(seed: int = 7) -> dict:
    """p0 is dealt A/B (1/2 each), acts x/y; p1 sees only p0's action and plays l/r; then p0
    acts again (u/d) remembering card, own action and p1's action. p0's second infosets are
    deeper in its own sequence, so this exercises the bottom-up order. Payoffs seeded ints."""
    rng = random.Random(seed)
    b = _Builder()
    deal = []
    for card in "AB":
        acts = []
        for a in "xy":
            p1_kids = []
            for r in "lr":
                last = b.decision(0, f"{card}{a}{r}", [(z, b.terminal(rng.randint(-5, 5)))
                                                       for z in "ud"])
                p1_kids.append((r, last))
            acts.append((a, b.decision(1, a, p1_kids)))
        deal.append((card, "1/2", b.decision(0, card, acts)))
    return b.game("two-stage", b.chance(deal))


def forgetful() -> dict:
    """p0 moves L/R, then moves again in ONE infoset that contains both resulting nodes:
    p0 forgets its own first move. Violates perfect recall."""
    b = _Builder()
    kids = []
    for a in "LR":
        kids.append((a, b.decision(0, "again", [("u", b.terminal(1)), ("d", b.terminal(-1))])))
    return b.game("forgetful", b.decision(0, "first", kids))


def absent_minded() -> dict:
    """The absent-minded driver: one p0 infoset holding a node and its own descendant."""
    b = _Builder()
    inner = b.decision(0, "drive", [("exit", b.terminal(4)), ("go", b.terminal(1))])
    return b.game("absent-minded", b.decision(0, "drive", [("exit", b.terminal(0)),
                                                          ("go", inner)]))


def uniform_profile(game: dict) -> dict:
    probs = {}
    for n in game["nodes"]:
        if n["kind"] == "decision":
            k = len(n["actions"])
            probs[n["infoset"]] = {a["label"]: f"1/{k}" for a in n["actions"]}
    return {"format": "bluff-strategy-v1", "game": game["name"], "probs": probs}


def random_profile(game: dict, rng: random.Random, pure_prob: float = 0.2) -> dict:
    """Random exact profile; some rows pure, some with zero entries, to hit reach-0 paths."""
    probs = {}
    for n in game["nodes"]:
        if n["kind"] != "decision" or n["infoset"] in probs:
            continue
        labels = [a["label"] for a in n["actions"]]
        if rng.random() < pure_prob:
            pick = rng.choice(labels)
            probs[n["infoset"]] = {a: ("1" if a == pick else "0") for a in labels}
            continue
        w = [rng.randint(0, 4) for _ in labels]
        if sum(w) == 0:
            w[0] = 1
        tot = sum(w)
        probs[n["infoset"]] = {a: str(Fraction(x, tot)) for a, x in zip(labels, w)}
    return {"format": "bluff-strategy-v1", "game": game["name"], "probs": probs}


def random_game(rng: random.Random, max_depth: int = 5, name: str = "random") -> dict:
    """A random game with perfect recall by construction.

    Each player's infoset label is the player's whole observation string, which contains every
    one of its own infosets and actions, so nodes sharing a label share the player's own
    sequence. Chance outcomes and actions are each seen by a random subset of players. The
    number of actions is a function of the label so every node in an infoset agrees.
    """
    b = _Builder()

    def build(depth: int, obs: tuple[str, str]) -> int:
        if depth >= max_depth or (depth > 0 and rng.random() < 0.25):
            return b.terminal(Fraction(rng.randint(-6, 6), rng.choice([1, 1, 2, 3])))
        if rng.random() < 0.3:
            k = rng.randint(2, 3)
            w = [rng.randint(1, 3) for _ in range(k)]
            tot = sum(w)
            seen = rng.choice([(True, True), (True, False), (False, True), (False, False)])
            outs = []
            for i in range(k):
                child_obs = tuple(o + (f"c{depth}{i};" if s else "") for o, s in zip(obs, seen))
                outs.append((f"c{i}", str(Fraction(w[i], tot)), build(depth + 1, child_obs)))
            return b.chance(outs)
        player = rng.randint(0, 1)
        label = f"P{player}[{obs[player]}]"
        k = 2 + (sum(map(ord, label)) % 2)
        other_sees = rng.random() < 0.5
        acts = []
        for i in range(k):
            child = list(obs)
            child[player] = obs[player] + f"{label}a{i};"
            if other_sees:
                child[1 - player] = obs[1 - player] + f"o{player}a{i};"
            acts.append((f"a{i}", build(depth + 1, tuple(child))))
        return b.decision(player, label, acts)

    root = build(0, ("", ""))
    return b.game(name, root)
