"""attempt-b tests: the compiled core against an independent brute force over pure strategies.

uv run pytest dafny/attempt-b/tests -q
"""
import glob
import itertools
import json
import os
import random
import sys
from fractions import Fraction

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.dirname(HERE))
import check  # noqa: E402

sys.setrecursionlimit(100_000)

MINE = sorted(glob.glob(os.path.join(HERE, "games", "*.game.json")))
REPO_SMALL = [os.path.join(ROOT, "games", "out", "kuhn.game.json")] + sorted(
    glob.glob(os.path.join(ROOT, "games", "out", "other", "*.game.json")))
PERFECT_RECALL = [g for g in MINE if "forgetful" not in g] + [g for g in REPO_SMALL if os.path.exists(g)]
BRUTE_LIMIT = 5000


def load(path):
    with open(path) as f:
        return json.load(f)


def infosets(game):
    out = {}
    for nd in game["nodes"]:
        if nd["kind"] == "decision":
            out[nd["infoset"]] = (nd["player"], [a["label"] for a in nd["actions"]])
    return out


def random_profile(game, rng):
    probs = {}
    for name, (_, labels) in infosets(game).items():
        w = [rng.randint(0, 5) for _ in labels]
        if sum(w) == 0:
            w[rng.randrange(len(w))] = 1
        probs[name] = {l: f"{x}/{sum(w)}" for l, x in zip(labels, w)}
    return {"format": "bluff-strategy-v1", "game": game["name"], "producer": "test", "probs": probs}


def value(game, prof, p, pure=None):
    """Expected utility to p; p plays `pure` (infoset -> label) when given, else the profile."""
    nodes = {nd["id"]: nd for nd in game["nodes"]}

    def v(i):
        nd = nodes[i]
        if nd["kind"] == "terminal":
            u = Fraction(nd["payoff_p0"])
            return u if p == 0 else -u
        if nd["kind"] == "chance":
            return sum(Fraction(o["prob"]) * v(o["child"]) for o in nd["outcomes"])
        if pure is not None and nd["player"] == p:
            return next(v(a["child"]) for a in nd["actions"] if a["label"] == pure[nd["infoset"]])
        row = prof["probs"][nd["infoset"]]
        return sum(Fraction(row[a["label"]]) * v(a["child"]) for a in nd["actions"])
    return v(game["root"])


def brute_br(game, prof, p):
    mine = [(n, labels) for n, (q, labels) in infosets(game).items() if q == p]
    count = 1
    for _, labels in mine:
        count *= len(labels)
    if count > BRUTE_LIMIT:
        return None
    return max(value(game, prof, p, dict(zip([n for n, _ in mine], choice)))
               for choice in itertools.product(*[labels for _, labels in mine]))


def run_core(tmp_path, game_path, prof):
    sp = tmp_path / "s.strategy.json"
    sp.write_text(json.dumps(prof))
    return check.run(game_path, str(sp))


@pytest.mark.parametrize("game_path", PERFECT_RECALL, ids=os.path.basename)
def test_core_matches_brute_force(tmp_path, game_path):
    game = load(game_path)
    rng = random.Random(2520)
    for _ in range(3):
        prof = random_profile(game, rng)
        v = run_core(tmp_path, game_path, prof)
        assert v["verified"] is True
        assert Fraction(v["value_p0"]) == value(game, prof, 0)
        for p in (0, 1):
            br = Fraction(v[f"br_value_p{p}"])
            # the returned pure best response attains the value (independent evaluation)
            assert value(game, prof, p, v[f"best_response_p{p}"]) == br
            bf = brute_br(game, prof, p)
            if bf is not None:
                assert bf == br
        nc = Fraction(v["br_value_p0"]) + Fraction(v["br_value_p1"])
        assert Fraction(v["nash_conv"]) == nc and nc >= 0
        assert Fraction(v["exploitability"]) == nc / 2


def test_kuhn_reference_equilibrium():
    g = os.path.join(ROOT, "games", "out", "kuhn.game.json")
    s = os.path.join(ROOT, "games", "out", "kuhn.ref-strategy.json")
    if not (os.path.exists(g) and os.path.exists(s)):
        pytest.skip("kuhn files not generated")
    v = check.run(g, s)
    assert v["value_p0"] == "-1/18" and v["exploitability"] == "0"


def test_imperfect_recall_is_refused(tmp_path):
    g = os.path.join(HERE, "games", "forgetful_b.game.json")
    prof = random_profile(load(g), random.Random(1))
    with pytest.raises(check.Malformed, match="not attained"):
        run_core(tmp_path, g, prof)


def test_bad_row_is_rejected(tmp_path):
    g = os.path.join(HERE, "games", "pennies_b.game.json")
    prof = {"format": "bluff-strategy-v1", "game": "pennies_b", "producer": "test",
            "probs": {"p0": {"H": "0.5", "T": "0.6"}, "p1": {"H": "1", "T": "0"}}}
    with pytest.raises(check.Malformed, match="sums to"):
        run_core(tmp_path, g, prof)


def test_float_rows_are_normalized(tmp_path):
    g = os.path.join(HERE, "games", "pennies_b.game.json")
    prof = {"format": "bluff-strategy-v1", "game": "pennies_b", "producer": "test",
            "probs": {"p0": {"H": "0.3333333", "T": "0.6666667"}, "p1": {"H": "0.5", "T": "0.5"}}}
    v = run_core(tmp_path, g, prof)
    assert v["value_p0"] == "0"
    # p1 best-responds to p0's (0.3333333, 0.6666667): playing H wins 2*0.6666667 - 1 for p1
    assert Fraction(v["br_value_p1"]) == Fraction(6666667 - 3333333, 10**7)


def test_wrong_json_shape_is_malformed(tmp_path):
    """A node that is not an object is malformed input (exit 2), not an internal crash."""
    g = load(os.path.join(HERE, "games", "pennies_b.game.json"))
    g["nodes"].append(5)
    gp = tmp_path / "bad.game.json"
    gp.write_text(json.dumps(g))
    prof = random_profile(load(os.path.join(HERE, "games", "pennies_b.game.json")), random.Random(3))
    with pytest.raises(check.Malformed, match="malformed input"):
        run_core(tmp_path, str(gp), prof)
