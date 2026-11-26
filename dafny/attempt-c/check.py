"""Thin trusted wrapper around the compiled Dafny core of attempt C.

Usage: uv run dafny/attempt-c/check.py GAME.game.json STRATEGY.strategy.json

Reads a bluff-game-v1 game and a bluff-strategy-v1 profile (CONTRACTS.md), builds the
Dafny values (infosets numbered in order of first appearance, actions in file order),
normalizes each strategy row exactly (CONTRACTS.md section 2), calls the verified
Checker.Check and prints a bluff-verdict-v1 JSON. Exit 2 on malformed input.

Everything in this file is in the trusted base: JSON parsing, rational parsing, the
tree construction, the row normalization and the printing of the result.
"""

import json
import sys
import threading
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUILD = HERE / "build" / "core-py"
sys.path.insert(0, str(BUILD))

import _dafny  # noqa: E402
import Checker  # noqa: E402
import Model  # noqa: E402

ROW_TOLERANCE = Fraction(1, 10**6)


class Malformed(Exception):
    pass


def rational(text, what):
    if not isinstance(text, str):
        raise Malformed(f"{what}: expected a rational string, got {text!r}")
    try:
        return Fraction(text.strip())
    except (ValueError, ZeroDivisionError) as exc:
        raise Malformed(f"{what}: not a rational: {text!r}") from exc


def dseq(items):
    return _dafny.SeqWithoutIsStrInference(list(items))


def dreal(q):
    return _dafny.BigRational(q)


def is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def list_of_dicts(x, what):
    if not isinstance(x, list) or not all(isinstance(e, dict) for e in x):
        raise Malformed(f"{what}: expected a list of objects")
    return x


def load_game(doc):
    if not isinstance(doc, dict) or doc.get("format") != "bluff-game-v1":
        raise Malformed("game: format must be bluff-game-v1")
    nodes = {}
    for node in list_of_dicts(doc.get("nodes"), "game: nodes"):
        nid = node.get("id")
        if not is_int(nid) or nid < 0 or nid in nodes:
            raise Malformed(f"game: bad or duplicate node id {nid!r}")
        nodes[nid] = node
    infosets = []  # names in order of first appearance
    index = {}
    labels = []  # action labels per infoset
    owner = []
    for node in doc["nodes"]:
        if node.get("kind") == "decision":
            name = node["infoset"]
            acts = [a["label"] for a in list_of_dicts(node["actions"], f"game: node {node['id']} actions")]
            player = node["player"]
            if not isinstance(name, str) or not all(isinstance(a, str) for a in acts):
                raise Malformed(f"game: node {node['id']} infoset and action labels must be strings")
            if len(set(acts)) != len(acts):
                raise Malformed(f"game: node {node['id']} has duplicate action labels {acts!r}")
            if not is_int(player) or player not in (0, 1):
                raise Malformed(f"game: node {node['id']} has player {player!r}")
            if name not in index:
                index[name] = len(infosets)
                infosets.append(name)
                labels.append(acts)
                owner.append(player)
            elif labels[index[name]] != acts or owner[index[name]] != player:
                raise Malformed(f"game: infoset {name!r} is inconsistent across its nodes")
    seen = set()

    def build(nid):
        if not is_int(nid) or nid not in nodes:
            raise Malformed(f"game: missing node {nid}")
        if nid in seen:
            raise Malformed(f"game: node {nid} reached twice (not a tree)")
        seen.add(nid)
        node = nodes[nid]
        kind = node.get("kind")
        if kind == "terminal":
            return Model.Node_Leaf(dreal(rational(node["payoff_p0"], f"node {nid} payoff")))
        if kind == "chance":
            outs = list_of_dicts(node["outcomes"], f"game: node {nid} outcomes")
            probs = [rational(o["prob"], f"node {nid} prob") for o in outs]
            if any(p <= 0 or p > 1 for p in probs) or sum(probs) != 1:
                raise Malformed(f"game: chance node {nid} probabilities are not a distribution")
            kids = [build(o["child"]) for o in outs]
            return Model.Node_Chance(dseq(dreal(p) for p in probs), dseq(kids))
        if kind == "decision":
            kids = [build(a["child"]) for a in node["actions"]]
            return Model.Node_Dec(node["player"], index[node["infoset"]], dseq(kids))
        raise Malformed(f"game: node {nid} has unknown kind {kind!r}")

    root = build(doc["root"])
    if len(seen) != len(nodes):
        raise Malformed("game: some nodes are unreachable from the root")
    game = Model.Game_Game(root, dseq(owner), dseq(len(a) for a in labels))
    return game, infosets, labels, owner


def load_profile(doc, infosets, labels):
    if not isinstance(doc, dict) or doc.get("format") != "bluff-strategy-v1":
        raise Malformed("strategy: format must be bluff-strategy-v1")
    probs = doc.get("probs")
    if not isinstance(probs, dict) or not all(isinstance(v, dict) for v in probs.values()):
        raise Malformed("strategy: probs must map infosets to objects")
    unknown = set(probs) - set(infosets)
    if unknown:
        raise Malformed(f"strategy: names infosets not in the game: {sorted(unknown)!r}")
    rows = []
    for name, acts in zip(infosets, labels):
        if name not in probs:
            raise Malformed(f"strategy: infoset {name!r} missing")
        given = probs[name]
        row = []
        for label in acts:
            if label not in given:
                raise Malformed(f"strategy: infoset {name!r} misses action {label!r}")
            q = rational(given[label], f"strategy {name}/{label}")
            if q < 0:
                raise Malformed(f"strategy: negative probability at {name}/{label}")
            row.append(q)
        if set(given) - set(acts):
            raise Malformed(f"strategy: infoset {name!r} has unknown actions")
        total = sum(row)
        if abs(total - 1) > ROW_TOLERANCE:
            raise Malformed(f"strategy: row {name!r} sums to {float(total)}, not 1 within 1e-6")
        rows.append([q / total for q in row])
    return dseq(dseq(dreal(q) for q in row) for row in rows)


def fmt(x):
    return str(Fraction(x))


def check(game_path, strategy_path):
    gdoc = json.loads(Path(game_path).read_text())
    sdoc = json.loads(Path(strategy_path).read_text())
    game, infosets, labels, owner = load_game(gdoc)
    profile = load_profile(sdoc, infosets, labels)
    result = Checker.default__.Check(game, profile)
    if result.is_Err:
        raise Malformed("core: " + "".join(result.msg))

    def br_map(pi, p):
        return {infosets[i]: labels[i][pi[i]] for i in range(len(infosets)) if owner[i] == p}

    expl = Fraction(result.expl)
    return {
        "format": "bluff-verdict-v1",
        "game": gdoc.get("name", ""),
        "value_p0": fmt(result.value__p0),
        "br_value_p0": fmt(result.br0),
        "br_value_p1": fmt(result.br1),
        "nash_conv": fmt(result.nash__conv),
        "exploitability": fmt(expl),
        "exploitability_float": float(expl),
        "best_response_p0": br_map(result.pi0, 0),
        "best_response_p1": br_map(result.pi1, 1),
        "verified": True,
    }


def main():
    if len(sys.argv) != 3:
        print("usage: check.py GAME.game.json STRATEGY.strategy.json", file=sys.stderr)
        sys.exit(2)
    out = {}

    def run():
        try:
            out["verdict"] = check(sys.argv[1], sys.argv[2])
        except (Malformed, KeyError, TypeError, ValueError, OSError) as exc:
            # json.JSONDecodeError and UnicodeDecodeError are ValueErrors.
            out["error"] = f"{type(exc).__name__}: {exc}"

    # The compiled core recurses along the game tree: give it a deep stack.
    sys.setrecursionlimit(1_000_000)
    threading.stack_size(512 * 1024 * 1024)
    worker = threading.Thread(target=run)
    worker.start()
    worker.join()
    if "error" in out:
        print(out["error"], file=sys.stderr)
        sys.exit(2)
    if "verdict" not in out:
        print("internal error: the checker thread ended without a verdict (see above)", file=sys.stderr)
        sys.exit(1)
    print(json.dumps(out["verdict"], indent=2))


if __name__ == "__main__":
    main()
