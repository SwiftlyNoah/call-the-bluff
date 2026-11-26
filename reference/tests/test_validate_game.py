import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import tiny_games as tg
from validate_game import GameError, load_game, validate_game_doc

HERE = Path(__file__).resolve().parent
SCRIPT = HERE.parent / "validate_game.py"


def reasons(doc) -> str:
    with pytest.raises(GameError) as e:
        validate_game_doc(doc)
    return " | ".join(e.value.reasons)


@pytest.mark.parametrize("make", [tg.matching_pennies, tg.bet_or_fold, tg.two_stage])
def test_hand_games_valid(make):
    g = validate_game_doc(make())
    assert g.infosets


def test_infosets_parsed():
    g = validate_game_doc(tg.bet_or_fold())
    assert g.infosets == {"H": (0, ["bet", "fold"]), "L": (0, ["bet", "fold"]),
                          "bet": (1, ["call", "fold"])}


def test_random_games_valid():
    import random
    rng = random.Random(1)
    for i in range(200):
        validate_game_doc(tg.random_game(rng, name=f"r{i}"))


def test_perfect_recall_violation_rejected():
    assert "perfect recall violated at infoset 'again'" in reasons(tg.forgetful())


def test_absent_minded_rejected():
    assert "perfect recall violated at infoset 'drive'" in reasons(tg.absent_minded())


def test_chance_sum_099_rejected():
    g = tg.bet_or_fold()
    root = next(n for n in g["nodes"] if n["id"] == g["root"])
    root["outcomes"][1]["prob"] = "0.65666"
    assert "not exactly 1" in reasons(g)


def test_chance_zero_prob_rejected():
    g = tg.bet_or_fold()
    root = next(n for n in g["nodes"] if n["id"] == g["root"])
    root["outcomes"][0]["prob"] = "0"
    root["outcomes"][1]["prob"] = "1"
    assert "not in (0, 1]" in reasons(g)


def test_float_in_game_rejected():
    g = tg.matching_pennies()
    g["nodes"][0]["payoff_p0"] = 1.0
    assert "schema" in reasons(g)


def test_exponent_rational_rejected():
    g = tg.matching_pennies()
    g["nodes"][0]["payoff_p0"] = "1e0"
    assert "schema" in reasons(g)


def test_duplicate_id_rejected():
    g = tg.matching_pennies()
    g["nodes"][1]["id"] = 0
    assert "duplicate node id 0" in reasons(g)


def test_shared_child_rejected():
    g = tg.matching_pennies()
    p1 = [n for n in g["nodes"] if n["kind"] == "decision" and n["player"] == 1]
    p1[1]["actions"][0]["child"] = p1[0]["actions"][0]["child"]
    r = reasons(g)
    assert "2 parents" in r and "0 parents" in r


def test_missing_child_rejected():
    g = tg.matching_pennies()
    g["nodes"][-1]["actions"][0]["child"] = 999
    assert "missing node 999" in reasons(g)


def test_unreachable_cycle_rejected():
    g = tg.matching_pennies()
    n = len(g["nodes"])
    g["nodes"].append({"id": n, "kind": "decision", "player": 0, "infoset": "loop",
                       "actions": [{"label": "a", "child": n + 1}]})
    g["nodes"].append({"id": n + 1, "kind": "decision", "player": 1, "infoset": "loop2",
                       "actions": [{"label": "a", "child": n}]})
    assert "unreachable from root" in reasons(g)


def test_root_missing_rejected():
    g = tg.matching_pennies()
    g["root"] = 77
    assert "root 77 is not a node" in reasons(g)


def test_inconsistent_actions_rejected():
    g = tg.matching_pennies()
    p1 = [n for n in g["nodes"] if n["kind"] == "decision" and n["player"] == 1]
    p1[1]["actions"].reverse()
    assert "infoset 'p1'" in reasons(g)


def test_infoset_across_players_rejected():
    g = tg.matching_pennies()
    p1 = [n for n in g["nodes"] if n["kind"] == "decision" and n["player"] == 1]
    for n in p1:
        n["infoset"] = "p0"
    assert "is player 1, others are player 0" in reasons(g) or \
        "is player 0, others are player 1" in reasons(g)


def test_duplicate_action_labels_rejected():
    g = tg.forgetful()
    root = next(n for n in g["nodes"] if n["id"] == g["root"])
    root["actions"][1]["label"] = "L"
    assert "duplicate edge labels" in reasons(g)


def test_known_value_requires_source():
    g = tg.matching_pennies()
    del g["known_value_source"]
    assert "known_value_source" in reasons(g)


def test_cli(tmp_path):
    good = tmp_path / "mp.game.json"
    good.write_text(json.dumps(tg.matching_pennies()))
    bad = tmp_path / "forget.game.json"
    bad.write_text(json.dumps(tg.forgetful()))
    r = subprocess.run([sys.executable, str(SCRIPT), str(good)], capture_output=True, text=True, check=False)
    assert r.returncode == 0 and "perfect recall holds" in r.stdout
    r = subprocess.run([sys.executable, str(SCRIPT), str(good), str(bad)],
                       capture_output=True, text=True, check=False)
    assert r.returncode == 2 and "perfect recall violated" in r.stderr


def test_duplicate_json_key_rejected(tmp_path):
    p = tmp_path / "dup.game.json"
    text = json.dumps(tg.matching_pennies())
    p.write_text(text.replace('"name": "matching-pennies"',
                              '"name": "matching-pennies", "name": "other"'))
    with pytest.raises(GameError, match="duplicate JSON key"):
        load_game(p)


def test_extra_field_rejected():
    g = copy.deepcopy(tg.matching_pennies())
    g["nodes"][0]["payoff_p1"] = "1"
    assert "schema" in reasons(g)


def test_tiny_chance_sum_error_rejected():
    # An exact check, not a tolerance: off by 1e-12 is still rejected.
    g = tg.bet_or_fold()
    root = next(n for n in g["nodes"] if n["id"] == g["root"])
    root["outcomes"][0]["prob"] = "0.333333333333"
    root["outcomes"][1]["prob"] = "0.666666666666"
    assert "not exactly 1" in reasons(g)


def test_integral_float_in_integer_field_rejected(tmp_path):
    # JSON Schema counts 1.0 as an integer; a lone p1 node with "player": 1.0 used to pass the
    # schema and crash the perfect-recall pass with a TypeError (exit 1, not 2). review-spec.
    doc = {"format": "bluff-game-v1", "name": "one", "root": 0, "nodes": [
        {"id": 0, "kind": "decision", "player": 0, "infoset": "a",
         "actions": [{"label": "x", "child": 1}, {"label": "y", "child": 2}]},
        {"id": 1, "kind": "decision", "player": 1, "infoset": "b",
         "actions": [{"label": "x", "child": 3}, {"label": "y", "child": 4}]},
        {"id": 2, "kind": "terminal", "payoff_p0": "0"},
        {"id": 3, "kind": "terminal", "payoff_p0": "1"},
        {"id": 4, "kind": "terminal", "payoff_p0": "-1"}]}
    validate_game_doc(copy.deepcopy(doc))
    p = tmp_path / "f.game.json"
    for old, new in [('"player": 1,', '"player": 1.0,'), ('"root": 0', '"root": 0.0'),
                     ('"child": 4', '"child": 4.0')]:
        text = json.dumps(doc)
        assert old in text
        p.write_text(text.replace(old, new))
        with pytest.raises(GameError, match="float literal"):
            load_game(p)
        r = subprocess.run([sys.executable, str(SCRIPT), str(p)], capture_output=True,
                           text=True, check=False)
        assert r.returncode == 2 and "float literal" in r.stderr
