import json
import random
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import jsonschema
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import tiny_games as tg
from br import best_response, brute_force_br, compute_verdict, expected_value
from fractions_util import normalize_strategy
from validate_game import validate_game_doc

HERE = Path(__file__).resolve().parent
SCRIPT = HERE.parent / "br.py"
VERDICT_SCHEMA = json.loads((HERE.parent.parent / "schema" / "verdict.schema.json").read_text())


def verdict_for(game_doc, probs, **kw):
    g = validate_game_doc(game_doc)
    sigma = normalize_strategy(g.infoset_actions(), probs)
    v = compute_verdict(g, sigma, **kw)
    jsonschema.validate(v, VERDICT_SCHEMA)
    return v


def test_matching_pennies_uniform_is_equilibrium():
    v = verdict_for(tg.matching_pennies(), tg.uniform_profile(tg.matching_pennies())["probs"])
    assert (v["value_p0"], v["br_value_p0"], v["br_value_p1"], v["nash_conv"],
            v["exploitability"]) == ("0", "0", "0", "0", "0")


def test_matching_pennies_hand_computed():
    # p0 H=0.7, p1 H=0.4. value = .28+.18-.42-.12 = -0.08; p0's best is T (+0.2);
    # p1's best against H=0.7 is T (+0.4). nash_conv 3/5, exploitability 3/10.
    v = verdict_for(tg.matching_pennies(), {"p0": {"H": "0.7", "T": "0.3"},
                                            "p1": {"H": "0.4", "T": "0.6"}})
    assert v["value_p0"] == "-2/25"
    assert v["br_value_p0"] == "1/5" and v["best_response_p0"] == {"p0": "T"}
    assert v["br_value_p1"] == "2/5" and v["best_response_p1"] == {"p1": "T"}
    assert v["nash_conv"] == "3/5" and v["exploitability"] == "3/10"
    assert v["exploitability_float"] == 0.3 and v["verified"] is False


def test_matching_pennies_pure_p0():
    v = verdict_for(tg.matching_pennies(), {"p0": {"H": "1", "T": "0"},
                                            "p1": {"H": "0.5", "T": "0.5"}})
    assert v["br_value_p1"] == "1" and v["best_response_p1"] == {"p1": "T"}
    assert v["br_value_p0"] == "0" and v["best_response_p0"] == {"p0": "H"}  # tie: first action
    assert v["exploitability"] == "1/2"


def test_bet_or_fold_hand_computed():
    # p0 bets H always, bets L half; p1 calls half. Worked by hand:
    # value = 1/3*(3/2) + 2/3*(1/2*(-1/2) - 1/2) = 0
    # BR p0 vs call 1/2: H bet (3/2 > -1), L bet (-1/2 > -1): 1/3*3/2 + 2/3*(-1/2) = 1/6
    # BR p1: reach(H,bet)=1/3, reach(L,bet)=1/3; call 1/3*(-2)+1/3*(2)=0 > fold -2/3,
    #        plus 2/3*1/2*(+1) from p0 folding L: 1/3. nash_conv 1/2.
    v = verdict_for(tg.bet_or_fold(), {"H": {"bet": "1", "fold": "0"},
                                       "L": {"bet": "1/2", "fold": "1/2"},
                                       "bet": {"call": "1/2", "fold": "1/2"}})
    assert v["value_p0"] == "0"
    assert v["br_value_p0"] == "1/6"
    assert v["best_response_p0"] == {"H": "bet", "L": "bet"}
    assert v["br_value_p1"] == "1/3" and v["best_response_p1"] == {"bet": "call"}
    assert v["nash_conv"] == "1/2" and v["exploitability"] == "1/4"


def test_bet_or_fold_equilibrium_from_indifference():
    # p0 always bets H, bets L with prob b; p1 calls with prob c.
    # p1 indifferent: call = (1/3)(-2) + (2/3)b(2) = fold = (1/3)(-1) + (2/3)b(-1) -> b = 1/6.
    # p0 at L indifferent: bet = c(-2) + (1-c)(1) = fold = -1 -> c = 2/3.
    # Value = 1/3*(2/3*2 + 1/3) + 2/3*(-1) = 5/9 - 2/3 = -1/9.
    v = verdict_for(tg.bet_or_fold(), {"H": {"bet": "1", "fold": "0"},
                                       "L": {"bet": "1/6", "fold": "5/6"},
                                       "bet": {"call": "2/3", "fold": "1/3"}})
    assert v["nash_conv"] == "0" and v["exploitability"] == "0"
    assert v["value_p0"] == "-1/9"


def test_unnormalized_float_rows_are_normalized_exactly():
    a = 1 / 3
    v = verdict_for(tg.matching_pennies(), {"p0": {"H": repr(a), "T": repr(1 - a)},
                                            "p1": {"H": "0.5", "T": "0.5000004"}})
    assert Fraction(v["nash_conv"]) >= 0


@pytest.mark.parametrize("make", [tg.matching_pennies, tg.bet_or_fold, tg.two_stage])
def test_brute_force_agrees_on_hand_games(make):
    rng = random.Random(0)
    doc = make()
    g = validate_game_doc(doc)
    for _ in range(25):
        sigma = normalize_strategy(g.infoset_actions(), tg.random_profile(doc, rng)["probs"])
        for p in (0, 1):
            v, s = best_response(g, sigma, p)
            bv, _ = brute_force_br(g, sigma, p)
            assert v == bv
            assert expected_value(g, {p: s, 1 - p: sigma}, p) == v


def test_brute_force_agrees_on_random_games():
    rng = random.Random(2520)
    checked = 0
    for i in range(300):
        doc = tg.random_game(rng, max_depth=5, name=f"r{i}")
        g = validate_game_doc(doc)
        sigma = normalize_strategy(g.infoset_actions(), tg.random_profile(doc, rng)["probs"])
        # compute_verdict raises CrossCheckError on any disagreement
        compute_verdict(g, sigma, brute=None, brute_limit=4096)
        checked += 1
    assert checked == 300


def test_two_stage_needs_bottom_up_order():
    # Against a profile where p1 always plays r, p0's best second move must be fixed before its
    # first move is judged; the brute force confirms the value.
    doc = tg.two_stage()
    probs = tg.uniform_profile(doc)["probs"]
    probs["x"] = {"l": "0", "r": "1"}
    probs["y"] = {"l": "1", "r": "0"}
    v = verdict_for(doc, probs, brute=True)
    g = validate_game_doc(doc)
    sigma = normalize_strategy(g.infoset_actions(), probs)
    assert Fraction(v["br_value_p0"]) == brute_force_br(g, sigma, 0)[0]


def run_cli(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)],
                          capture_output=True, text=True, check=False)


def test_cli_outputs_verdict(tmp_path):
    gp = tmp_path / "bof.game.json"
    gp.write_text(json.dumps(tg.bet_or_fold()))
    sp = tmp_path / "bof.strategy.json"
    sp.write_text(json.dumps(tg.uniform_profile(tg.bet_or_fold())))
    r = run_cli(gp, sp, "--brute")
    assert r.returncode == 0, r.stderr
    v = json.loads(r.stdout)
    jsonschema.validate(v, VERDICT_SCHEMA)
    assert "brute force over" in r.stderr


def test_cli_rejects_row_summing_to_099(tmp_path):
    gp = tmp_path / "mp.game.json"
    gp.write_text(json.dumps(tg.matching_pennies()))
    sp = tmp_path / "mp.strategy.json"
    sp.write_text(json.dumps({"format": "bluff-strategy-v1", "game": "matching-pennies",
                              "probs": {"p0": {"H": "0.5", "T": "0.49"},
                                        "p1": {"H": "0.5", "T": "0.5"}}}))
    r = run_cli(gp, sp)
    assert r.returncode == 2 and "not 1 within 1e-6" in r.stderr and r.stdout == ""


def test_cli_rejects_float_probs_and_wrong_game(tmp_path):
    gp = tmp_path / "mp.game.json"
    gp.write_text(json.dumps(tg.matching_pennies()))
    sp = tmp_path / "s.strategy.json"
    sp.write_text(json.dumps({"format": "bluff-strategy-v1", "game": "matching-pennies",
                              "probs": {"p0": {"H": 0.5, "T": "0.5"},
                                        "p1": {"H": "0.5", "T": "0.5"}}}))
    assert run_cli(gp, sp).returncode == 2
    doc = tg.uniform_profile(tg.matching_pennies())
    doc["game"] = "kuhn"
    sp.write_text(json.dumps(doc))
    r = run_cli(gp, sp)
    assert r.returncode == 2 and "kuhn" in r.stderr


def test_cli_rejects_bad_game(tmp_path):
    gp = tmp_path / "f.game.json"
    gp.write_text(json.dumps(tg.forgetful()))
    sp = tmp_path / "f.strategy.json"
    sp.write_text(json.dumps(tg.uniform_profile(tg.forgetful())))
    r = run_cli(gp, sp)
    assert r.returncode == 2 and "perfect recall" in r.stderr


KUHN = HERE.parent.parent / "games" / "out" / "kuhn.game.json"


@pytest.mark.skipif(not KUHN.exists(), reason="games/out/kuhn.game.json not generated yet")
def test_kuhn_equilibrium_alpha_one_sixth():
    # Kuhn (1950) equilibrium family at alpha = 1/6, written by hand in tests/data.
    r = run_cli(KUHN, HERE / "data" / "kuhn_eq_alpha16.strategy.json", "--brute")
    assert r.returncode == 0, r.stderr
    v = json.loads(r.stdout)
    assert v["value_p0"] == "-1/18"
    assert v["nash_conv"] == "0" and v["exploitability"] == "0"


GAMES_OUT = HERE.parent.parent / "games" / "out"


@pytest.mark.skipif(not (GAMES_OUT / "leduc.game.json").exists(), reason="games not generated")
def test_agrees_with_lp_oracle_best_response():
    # Differential check against reference/lp.py's own exact best response (a separate,
    # top-down implementation) on Kuhn and Leduc, uniform and random profiles. review-spec.
    lp = pytest.importorskip("lp")
    for name in ("kuhn", "leduc"):
        path = GAMES_OUT / f"{name}.game.json"
        g, lg = validate_game_doc(json.loads(path.read_text())), lp.load_game(path)
        doc = json.loads(path.read_text())
        rng = random.Random(5)
        profiles = [tg.uniform_profile(doc)] + [tg.random_profile(doc, rng) for _ in range(2)]
        for prof in profiles:
            sigma = normalize_strategy(g.infoset_actions(), prof["probs"])
            for p in (0, 1):
                assert best_response(g, sigma, p)[0] == lp.best_response_value(lg, sigma, p)[0]
