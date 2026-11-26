"""Red-team fuzz of the gap between the Dafny model and what a game file means.

Usage: uv run dafny/core/tests/fuzz.py [--n N] [--seed S] [--mode pr|ir|both]

Generates random small games with nested chance (chance under decisions and under chance),
infosets spanning different depths, per-infoset permuted action orders, many infosets, chance
outcomes with probability 1, and strategies with zero-probability rows. In mode "pr" the
infoset key includes the player's own action sequence (perfect recall by construction); in
mode "ir" it does not (imperfect recall, absentmindedness possible).

For each (game, strategy) it runs the checker in process (dafny/core/adapter.run, the same code
bin/bluff runs) and compares against an independent brute force written here: the maximum over
every pure strategy of each player, value_p0 by direct recursion. For perfect-recall games it
also runs reference/br.py's compute_verdict with brute=True. Any mismatch is printed and the
game and strategy are written to dafny/core/tests/cases/.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
import threading
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "dafny" / "core"))
sys.path.insert(0, str(REPO / "reference"))
import adapter  # noqa: E402
import br as refbr  # noqa: E402
from validate_game import validate_game_doc, GameError  # noqa: E402
from fractions_util import normalize_strategy  # noqa: E402

LABELS = ["a", "b", "c", "x", "y", "z"]


def fmt(q: Fraction) -> str:
    return str(q.numerator) if q.denominator == 1 else f"{q.numerator}/{q.denominator}"


def gen_game(rng: random.Random, pr: bool, name: str) -> dict:
    nodes: list[dict] = []
    infosets: dict[tuple, tuple[str, list[str]]] = {}
    max_depth = rng.randint(2, 5)
    nbuckets = rng.randint(1, 3)

    def new_id() -> int:
        nodes.append({})
        return len(nodes) - 1

    def build(depth: int, seqs: tuple, signal: int) -> int:
        nid = new_id()
        r = rng.random()
        if depth >= max_depth or (depth > 0 and r < 0.15):
            nodes[nid] = {"id": nid, "kind": "terminal",
                          "payoff_p0": fmt(Fraction(rng.randint(-6, 6), rng.choice([1, 1, 2, 3])))}
            return nid
        if r < 0.35:
            k = rng.randint(1, 3)
            w = [rng.randint(1, 4) for _ in range(k)]
            t = sum(w)
            outs = []
            for i in range(k):
                c = build(depth + 1, seqs, rng.randrange(nbuckets))
                outs.append({"label": f"o{i}", "prob": fmt(Fraction(w[i], t)), "child": c})
            nodes[nid] = {"id": nid, "kind": "chance", "outcomes": outs}
            return nid
        q = rng.randint(0, 1)
        key = (q, seqs[q] if pr else (), signal % nbuckets)
        if key not in infosets:
            k = rng.randint(1, 3)
            labs = rng.sample(LABELS, k)
            infosets[key] = (f"P{q}|{len(infosets)}", labs)
        iname, labs = infosets[key]
        acts = []
        for lab in labs:
            ns = list(seqs)
            ns[q] = seqs[q] + ((iname, lab),)
            c = build(depth + 1, tuple(ns), signal if rng.random() < 0.7 else rng.randrange(nbuckets))
            acts.append({"label": lab, "child": c})
        nodes[nid] = {"id": nid, "kind": "decision", "player": q, "infoset": iname, "actions": acts}
        return nid

    root = build(0, ((), ()), 0)
    # shuffle the node list order, keep ids
    rng.shuffle(nodes)
    return {"format": "bluff-game-v1", "name": name, "description": "fuzz", "root": root, "nodes": nodes}


def infoset_table(game: dict) -> dict[str, tuple[int, list[str]]]:
    out = {}
    for n in game["nodes"]:
        if n["kind"] == "decision":
            out.setdefault(n["infoset"], (n["player"], [a["label"] for a in n["actions"]]))
    return out


def gen_strategy(rng: random.Random, game: dict) -> dict:
    probs = {}
    for name, (_, acts) in infoset_table(game).items():
        mode = rng.random()
        if mode < 0.3:
            w = [0] * len(acts)
            w[rng.randrange(len(acts))] = 1
        else:
            w = [rng.choice([0, 0, 1, 2, 3, 5]) for _ in acts]
            if sum(w) == 0:
                w[rng.randrange(len(w))] = 1
        t = sum(w)
        probs[name] = {a: fmt(Fraction(x, t)) for a, x in zip(acts, w)}
    return {"format": "bluff-strategy-v1", "game": game["name"], "producer": "redteam-model fuzz", "probs": probs}


# ------------------------------------------------------------ independent semantics


def ev(game: dict, plans: dict, p: int) -> Fraction:
    """Expected utility to p. plans[q][infoset] is a dict action->prob or a str (pure)."""
    by_id = {n["id"]: n for n in game["nodes"]}

    def go(nid: int) -> Fraction:
        n = by_id[nid]
        if n["kind"] == "terminal":
            x = Fraction(n["payoff_p0"])
            return x if p == 0 else -x
        if n["kind"] == "chance":
            return sum((Fraction(o["prob"]) * go(o["child"]) for o in n["outcomes"]), Fraction(0))
        plan = plans[n["player"]][n["infoset"]]
        if isinstance(plan, str):
            return next(go(a["child"]) for a in n["actions"] if a["label"] == plan)
        return sum((plan[a["label"]] * go(a["child"]) for a in n["actions"]), Fraction(0))

    return go(game["root"])


def brute(game: dict, sigma: dict, p: int) -> Fraction:
    tab = infoset_table(game)
    mine = sorted(i for i, (q, _) in tab.items() if q == p)
    best = None
    for combo in itertools.product(*(tab[i][1] for i in mine)):
        pure = dict(zip(mine, combo))
        v = ev(game, {p: pure, 1 - p: sigma}, p)
        if best is None or v > best:
            best = v
    return best


def has_perfect_recall(game: dict) -> bool:
    try:
        validate_game_doc(json.loads(json.dumps(game)))
        return True
    except GameError:
        return False


def run_case(game: dict, strat: dict, check_ref: bool) -> list[str]:
    problems = []
    tab = infoset_table(game)
    sigma = {i: {a: Fraction(x) for a, x in row.items()} for i, row in strat["probs"].items()}
    # exact rows already sum to 1, so normalization is the identity here
    D, C, G = adapter.load_core()
    try:
        root, arity, names, labels, owner = adapter.build_game(json.loads(json.dumps(game)), D, G)
        prof = adapter.build_profile(strat, game["name"], names, labels, D)
    except adapter.Malformed as e:
        return [f"adapter rejected a well-formed fuzz game: {e}"]
    res = C.default__.Check(C.GameT_GameT(root, D.SeqWithoutIsStrInference(arity)), prof)
    pr = has_perfect_recall(game)
    if res.is_Fail:
        if pr:
            problems.append("core refused a perfect-recall game: " + "".join(res.reason))
        return problems
    v = res.verdict
    truth = {
        "value_p0": ev(game, {0: sigma, 1: sigma}, 0),
        "br_value_p0": brute(game, sigma, 0),
        "br_value_p1": brute(game, sigma, 1),
    }
    got = {"value_p0": Fraction(v.value__p0), "br_value_p0": Fraction(v.br__value__p0),
           "br_value_p1": Fraction(v.br__value__p1)}
    for k in truth:
        if truth[k] != got[k]:
            problems.append(f"{k}: checker {fmt(got[k])}, brute {fmt(truth[k])} (perfect recall {pr})")
    # the returned pure BRs attain
    for p, br in ((0, v.br0), (1, v.br1)):
        pure = {names[I]: labels[I][br[I]] for I in range(len(names)) if owner[I] == p}
        att = ev(game, {p: pure, 1 - p: sigma}, p)
        if att != got[f"br_value_p{p}"]:
            problems.append(f"p{p} returned BR attains {fmt(att)} not {fmt(got[f'br_value_p{p}'])}")
    if check_ref and pr:
        g = validate_game_doc(json.loads(json.dumps(game)))
        nsig = normalize_strategy(g.infoset_actions(), strat["probs"])
        rv = refbr.compute_verdict(g, nsig, brute=True, brute_limit=10**6)
        for k in truth:
            if Fraction(rv[k]) != got[k]:
                problems.append(f"{k}: checker {fmt(got[k])}, reference/br.py {rv[k]}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=["pr", "ir", "both"], default="both")
    ap.add_argument("--shuffle-rows", action="store_true",
                    help="write each strategy row's keys in a random order (and the infosets too)")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    modes = ["pr", "ir"] if args.mode == "both" else [args.mode]
    stats = {"cases": 0, "problems": 0, "ir_games": 0, "ir_accepted": 0,
             "max_nodes": 0, "max_infosets": 0}
    outdir = HERE / "cases"
    for i in range(args.n):
        for m in modes:
            game = gen_game(rng, m == "pr", f"fuzz-{m}-{args.seed}-{i}")
            tab = infoset_table(game)
            npure = [1, 1]
            for q, acts in tab.values():
                npure[q] *= len(acts)
            if max(npure) > 5000:
                continue
            strat = gen_strategy(rng, game)
            if args.shuffle_rows:
                items = list(strat["probs"].items())
                rng.shuffle(items)
                strat["probs"] = {k: dict(rng.sample(list(r.items()), len(r))) for k, r in items}
            stats["cases"] += 1
            stats["max_nodes"] = max(stats["max_nodes"], len(game["nodes"]))
            stats["max_infosets"] = max(stats["max_infosets"], len(tab))
            pr = has_perfect_recall(game)
            if not pr:
                stats["ir_games"] += 1
            probs = run_case(game, strat, check_ref=True)
            if probs:
                stats["problems"] += 1
                outdir.mkdir(exist_ok=True)
                (outdir / f"{game['name']}.game.json").write_text(json.dumps(game, indent=1))
                (outdir / f"{game['name']}.strategy.json").write_text(json.dumps(strat, indent=1))
                print(game["name"], probs)
            if not pr:
                # did the core accept it?
                D, C, G = adapter.load_core()
                root, arity, names, labels, owner = adapter.build_game(json.loads(json.dumps(game)), D, G)
                prof = adapter.build_profile(strat, game["name"], names, labels, D)
                if C.default__.Check(C.GameT_GameT(root, D.SeqWithoutIsStrInference(arity)), prof).is_Ok:
                    stats["ir_accepted"] += 1
    print(json.dumps(stats))
    return 1 if stats["problems"] else 0


if __name__ == "__main__":
    sys.setrecursionlimit(100000)
    threading.stack_size(1 << 27)
    out = {}
    t = threading.Thread(target=lambda: out.setdefault("rc", main()))
    t.start()
    t.join()
    sys.exit(out.get("rc", 1))
