"""Differential test: each game file against its direct rules simulator (referees.py).

Two passes per game:
  - exhaustive: every node of the game file is visited, and the referee is replayed along the same
    history; at every node the kind, acting player, legal action labels (in order), chance labels
    and exact probabilities, and terminal payoffs must agree.
  - random playouts: histories sampled with random chance outcomes and random actions (the way the
    poker task tests Kuhn and Leduc), compared step by step the same way.
In both passes the infosets of the game file must induce exactly the same partition of histories as
the referee's observations: one infoset <-> one (player, observation), both ways.

Run from the repo root:  uv run python games/other/differential.py [--playouts N] [--seed S]
"""

from __future__ import annotations

import argparse
import random
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gamelib import OUT_DIR, by_id, load_game
from referees import REFEREES


class Mismatch(AssertionError):
    pass


def _replay(factory, prefix: list[str]):
    """Runs a fresh referee through `prefix`; returns ("request", req) or ("done", payoff)."""
    gen = factory()()
    try:
        req = next(gen)
        for choice in prefix:
            req = gen.send(choice)
    except StopIteration as stop:
        return "done", stop.value
    return "request", req


class Checker:
    def __init__(self, game: dict, factory) -> None:
        self.game = game
        self.nodes = by_id(game)
        self.factory = factory
        self.iset_to_obs: dict[str, tuple] = {}
        self.obs_to_iset: dict[tuple, str] = {}
        self.steps = 0

    def compare(self, nid: int, prefix: list[str]) -> None:
        self.steps += 1
        n = self.nodes[nid]
        status, got = _replay(self.factory, prefix)
        where = f"{self.game['name']} node {nid} history {prefix}"
        if n["kind"] == "terminal":
            if status != "done":
                raise Mismatch(f"{where}: tree ends, referee asks {got}")
            if Fraction(got) != Fraction(n["payoff_p0"]):
                raise Mismatch(f"{where}: payoff tree {n['payoff_p0']} referee {got}")
            return
        if status == "done":
            raise Mismatch(f"{where}: referee ended with {got}, tree has a {n['kind']} node")
        if n["kind"] == "chance":
            if got[0] != "chance":
                raise Mismatch(f"{where}: tree chance, referee {got[0]}")
            tree = {o["label"]: Fraction(o["prob"]) for o in n["outcomes"]}
            ref = {k: Fraction(v) for k, v in got[1].items()}
            if tree != ref:
                raise Mismatch(f"{where}: chance tree {tree} referee {ref}")
            return
        if got[0] != "move":
            raise Mismatch(f"{where}: tree decision, referee {got[0]}")
        _, player, obs, legal = got
        if player != n["player"]:
            raise Mismatch(f"{where}: player tree {n['player']} referee {player}")
        labels = tuple(a["label"] for a in n["actions"])
        if labels != tuple(legal):
            raise Mismatch(f"{where}: actions tree {labels} referee {legal}")
        key = (player, obs)
        iset = n["infoset"]
        if self.iset_to_obs.setdefault(iset, key) != key:
            raise Mismatch(f"{where}: infoset {iset!r} covers two observations {self.iset_to_obs[iset]} and {key}")
        if self.obs_to_iset.setdefault(key, iset) != iset:
            raise Mismatch(f"{where}: observation {key} split over infosets {self.obs_to_iset[key]!r} and {iset!r}")

    def children(self, nid: int):
        n = self.nodes[nid]
        if n["kind"] == "chance":
            return [(o["label"], o["child"], Fraction(o["prob"])) for o in n["outcomes"]]
        if n["kind"] == "decision":
            return [(a["label"], a["child"], None) for a in n["actions"]]
        return []

    def exhaustive(self) -> int:
        stack = [(self.game["root"], [])]
        visited = 0
        while stack:
            nid, prefix = stack.pop()
            self.compare(nid, prefix)
            visited += 1
            for label, child, _ in self.children(nid):
                stack.append((child, prefix + [label]))
        if visited != len(self.nodes):
            raise Mismatch(f"{self.game['name']}: visited {visited} of {len(self.nodes)} nodes")
        return visited

    def random_playouts(self, count: int, rng: random.Random) -> int:
        for _ in range(count):
            nid, prefix = self.game["root"], []
            while True:
                self.compare(nid, prefix)
                kids = self.children(nid)
                if not kids:
                    break
                if kids[0][2] is not None:  # chance: sample by the tree's own probabilities
                    weights = [float(p) for _, _, p in kids]
                    label, child, _ = rng.choices(kids, weights=weights)[0]
                else:
                    label, child, _ = rng.choice(kids)
                nid, prefix = child, prefix + [label]
        return count


def check_file(path: Path, playouts: int, seed: int) -> dict:
    game = load_game(path)
    checker = Checker(game, REFEREES[game["name"]])
    visited = checker.exhaustive()
    checker.random_playouts(playouts, random.Random(seed))
    return {
        "game": game["name"],
        "nodes_checked": visited,
        "playouts": playouts,
        "infosets_matched": len(checker.iset_to_obs),
        "steps": checker.steps,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--playouts", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=2520)
    args = ap.parse_args()
    for path in sorted(OUT_DIR.glob("*.game.json")):
        r = check_file(path, args.playouts, args.seed)
        print(
            f"{r['game']}: OK, all {r['nodes_checked']} nodes match the referee, "
            f"{r['playouts']} random playouts match, {r['infosets_matched']} infosets <-> observations 1:1"
        )


if __name__ == "__main__":
    main()
