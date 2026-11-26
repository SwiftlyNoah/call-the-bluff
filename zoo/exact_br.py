"""The zoo's own exact best response (fractions), used when reference/br.py is absent and as a
second opinion when it is present. It shares no code with reference/ or with zoo/cfr.py.

    uv run zoo/exact_br.py GAME.game.json STRATEGY.strategy.json   # prints exploitability JSON
"""

from __future__ import annotations

import json
import sys
from fractions import Fraction


def load(game_path: str, strat_path: str) -> tuple[dict, dict[str, list[Fraction]]]:
    with open(game_path) as f:
        game = json.load(f)
    with open(strat_path) as f:
        probs = json.load(f)["probs"]
    nodes = {n["id"]: n for n in game["nodes"]}
    sigma: dict[str, list[Fraction]] = {}
    for n in nodes.values():
        if n["kind"] == "decision" and n["infoset"] not in sigma:
            row = [Fraction(probs[n["infoset"]][a["label"]]) for a in n["actions"]]
            s = sum(row)
            if abs(s - 1) > Fraction(1, 10**6):
                raise ValueError(f"row {n['infoset']} sums to {float(s)}")
            sigma[n["infoset"]] = [x / s for x in row]
    return game, sigma


def br_value(game: dict, sigma: dict[str, list[Fraction]], p: int) -> Fraction:
    nodes = {n["id"]: n for n in game["nodes"]}
    w: dict[int, Fraction] = {}
    members: dict[str, list[int]] = {}
    stack = [(game["root"], Fraction(1))]
    while stack:
        nid, r = stack.pop()
        w[nid] = r
        n = nodes[nid]
        if n["kind"] == "chance":
            stack += [(o["child"], r * Fraction(o["prob"])) for o in n["outcomes"]]
        elif n["kind"] == "decision":
            if n["player"] == p:
                members.setdefault(n["infoset"], []).append(nid)
                stack += [(a["child"], r) for a in n["actions"]]
            else:
                stack += [(a["child"], r * q) for a, q in zip(n["actions"], sigma[n["infoset"]])]
    choice: dict[str, int] = {}
    memo: dict[int, Fraction] = {}

    def val(nid: int) -> Fraction:
        if nid in memo:
            return memo[nid]
        n = nodes[nid]
        if n["kind"] == "terminal":
            u = Fraction(n["payoff_p0"])
            v = u if p == 0 else -u
        elif n["kind"] == "chance":
            v = sum((Fraction(o["prob"]) * val(o["child"]) for o in n["outcomes"]), Fraction(0))
        elif n["player"] == p:
            k = n["infoset"]
            if k not in choice:
                tots = [sum((w[h] * val(nodes[h]["actions"][a]["child"]) for h in members[k]), Fraction(0))
                        for a in range(len(n["actions"]))]
                choice[k] = max(range(len(tots)), key=tots.__getitem__)
            v = val(n["actions"][choice[k]]["child"])
        else:
            v = sum((q * val(a["child"]) for a, q in zip(n["actions"], sigma[n["infoset"]])), Fraction(0))
        memo[nid] = v
        return v

    return val(game["root"])


def exploitability(game_path: str, strat_path: str) -> Fraction:
    game, sigma = load(game_path, strat_path)
    return (br_value(game, sigma, 0) + br_value(game, sigma, 1)) / 2


if __name__ == "__main__":
    sys.setrecursionlimit(100000)
    e = exploitability(sys.argv[1], sys.argv[2])
    print(json.dumps({"exploitability": f"{e.numerator}/{e.denominator}", "exploitability_float": float(e)}))
