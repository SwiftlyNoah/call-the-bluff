"""Fast checks on the zoo: the clean solver solves Kuhn, every mutant edit applies exactly once,
and the zoo's exact BR agrees with the LP oracle's equilibrium (exploitability exactly 0).

    uv run pytest zoo/tests -q
"""

from __future__ import annotations

import json
import sys
from fractions import Fraction
from pathlib import Path

ZOO = Path(__file__).resolve().parent.parent
ROOT = ZOO.parent
sys.path.insert(0, str(ZOO))

import cfr
import exact_br
import fallback_games
import make_mutants


def kuhn_game() -> dict:
    return fallback_games.kuhn()


def test_clean_cfrplus_solves_kuhn() -> None:
    _, meters = cfr.solve(kuhn_game(), "cfrplus", 3000, 100)
    assert meters["converged"]
    assert meters["self_exploitability"] < 1e-3
    assert abs(meters["self_value_p0"] - (-1 / 18)) < 1e-3


def test_every_mutant_edit_matches_exactly_once() -> None:
    clean = (ZOO / "cfr.py").read_text()
    for name, (old, new) in make_mutants.MUTANTS.items():
        assert clean.count(old) == 1, name
        assert old != new, name


def test_generated_mutants_are_current() -> None:
    clean = (ZOO / "cfr.py").read_text()
    for name, (old, new) in make_mutants.MUTANTS.items():
        path = ZOO / "mutants" / f"{name}.py"
        body = path.read_text().split("\n", 1)[1]
        assert body == clean.replace(old, new), f"{name}: rerun uv run zoo/make_mutants.py"


def test_exact_br_zero_on_lp_equilibrium(tmp_path: Path) -> None:
    game = ROOT / "games/out/kuhn.game.json"
    strat = ROOT / "games/out/kuhn.lp-strategy.json"
    if not (game.exists() and strat.exists()):
        game = tmp_path / "kuhn.game.json"
        game.write_text(json.dumps(kuhn_game()))
        a = Fraction(1, 6)  # a Kuhn equilibrium with alpha = 1/6 (Kuhn 1950)
        probs = {
            "J:": {"check": 1 - a, "bet": a}, "Q:": {"check": 1, "bet": 0}, "K:": {"check": 1 - 3 * a, "bet": 3 * a},
            "J:cb": {"fold": 1, "call": 0}, "Q:cb": {"fold": Fraction(2, 3) - a, "call": a + Fraction(1, 3)},
            "K:cb": {"fold": 0, "call": 1},
            "J:c": {"check": Fraction(2, 3), "bet": Fraction(1, 3)}, "Q:c": {"check": 1, "bet": 0},
            "K:c": {"check": 0, "bet": 1}, "J:b": {"fold": 1, "call": 0},
            "Q:b": {"fold": Fraction(2, 3), "call": Fraction(1, 3)}, "K:b": {"fold": 0, "call": 1},
        }
        strat = tmp_path / "kuhn.strategy.json"
        strat.write_text(json.dumps({"format": "bluff-strategy-v1", "game": "kuhn", "producer": "test",
                                     "probs": {k: {a: str(v) for a, v in r.items()} for k, r in probs.items()}}))
    assert exact_br.exploitability(str(game), str(strat)) == 0


def _true_exploitability(solver, game: dict, tmp_path: Path, name: str) -> float:
    """Grade a solver's shipped strategy with the zoo's exact BR (not with its own meter)."""
    gpath = tmp_path / f"{name}.game.json"
    gpath.write_text(json.dumps(game))
    spath = tmp_path / f"{name}.strategy.json"
    spath.write_text(json.dumps({"format": "bluff-strategy-v1", "game": game["name"], "producer": "test",
                                 "probs": solver.strategy_probs()}))
    return float(exact_br.exploitability(str(gpath), str(spath)))


def _run_mutant(name: str, game: dict, algo: str = "cfrplus"):
    src = (ZOO / "mutants" / f"{name}.py").read_text()
    ns: dict = {"__name__": "mutant"}
    exec(compile(src, f"{name}.py", "exec"), ns)  # noqa: S102
    return ns["solve"](game, algo, 3000, 100)


def test_clean_meter_is_honest_on_kuhn(tmp_path: Path) -> None:
    # The control: for the clean solver, the self-BR meter and the exact grade agree.
    game = kuhn_game()
    s, meters = cfr.solve(game, "cfrplus", 3000, 100)
    true = _true_exploitability(s, game, tmp_path, "clean")
    assert true < 1e-3
    assert abs(true - meters["self_exploitability"]) < 1e-9


def test_info_leak_fools_its_own_meter_on_kuhn(tmp_path: Path) -> None:
    # Fooled means BOTH halves: the meter says converged AND the shipped strategy is bad.
    # (Checking only the meter would pass for the clean solver too.)
    game = kuhn_game()
    s, meters = _run_mutant("info_leak_key", game)
    assert meters["converged"]
    assert meters["self_exploitability"] < 1e-3
    assert _true_exploitability(s, game, tmp_path, "leak") > 0.1


def test_current_not_average_fools_its_own_meter_on_kuhn(tmp_path: Path) -> None:
    game = kuhn_game()
    s, meters = _run_mutant("current_not_average", game, "cfr")
    assert meters["converged"]
    assert _true_exploitability(s, game, tmp_path, "cur") > 1e-2


def test_chance_ignored_fools_its_own_meter_on_nonuniform_chance(tmp_path: Path) -> None:
    # A two-card toy with a 3/4 vs 1/4 deal: p0 sees its card and bets or checks; p1 sees only
    # the bet and calls or folds. A solver that treats the deal as 50/50 solves the wrong game.
    def dec(pl, iset, acts):
        return {"kind": "decision", "player": pl, "infoset": iset, "actions": acts}
    nodes = [
        {"id": 0, "kind": "chance", "outcomes": [{"label": "H", "prob": "3/4", "child": 1},
                                                 {"label": "L", "prob": "1/4", "child": 2}]},
        dict(dec(0, "H", [{"label": "check", "child": 3}, {"label": "bet", "child": 4}]), id=1),
        dict(dec(0, "L", [{"label": "check", "child": 5}, {"label": "bet", "child": 6}]), id=2),
        {"id": 3, "kind": "terminal", "payoff_p0": "1"},
        dict(dec(1, "b", [{"label": "fold", "child": 7}, {"label": "call", "child": 8}]), id=4),
        {"id": 5, "kind": "terminal", "payoff_p0": "-1"},
        dict(dec(1, "b", [{"label": "fold", "child": 9}, {"label": "call", "child": 10}]), id=6),
        {"id": 7, "kind": "terminal", "payoff_p0": "1"},
        {"id": 8, "kind": "terminal", "payoff_p0": "2"},
        {"id": 9, "kind": "terminal", "payoff_p0": "1"},
        {"id": 10, "kind": "terminal", "payoff_p0": "-2"},
    ]
    game = {"format": "bluff-game-v1", "name": "skewed", "root": 0, "nodes": nodes}
    s, meters = cfr.solve(game, "cfrplus", 3000, 100)
    assert _true_exploitability(s, game, tmp_path, "clean") < 1e-2
    s, meters = _run_mutant("chance_ignored", game)
    assert meters["converged"]
    assert _true_exploitability(s, game, tmp_path, "chance") > 1e-2


def _redblack(name: str) -> dict | None:
    path = ROOT / "games/out/redblack" / f"{name}.game.json"
    return json.loads(path.read_text()) if path.exists() else None


def test_color_swap_is_an_automorphism_of_a_dealt_game() -> None:
    # The swap the clean meter uses maps every infoset of a 26/26 deal to an infoset of the same
    # player with the swapped actions, and is an involution.
    game = _redblack("rb-21")
    if game is None:
        return
    labels = {n["infoset"]: (n["player"], sorted(a["label"] for a in n["actions"]))
              for n in game["nodes"] if n["kind"] == "decision"}
    for i, (pl, acts) in labels.items():
        j = cfr.color_swap(i)
        assert cfr.color_swap(j) == i
        assert labels[j] == (pl, sorted(cfr.color_swap(a) for a in acts)), i
    assert cfr.color_swap("p0|deal:RB r1:s0 ro:BR 0:2R") == "p0|deal:RB r1:s0 ro:RB 0:2B"
    assert cfr.color_swap("J:cb") == "J:cb"


def test_color_swap_imposed_fools_its_own_meter_on_the_gate_endgame(tmp_path: Path) -> None:
    # Imposed where it holds (a dealt game) the swap is harmless; imposed on the gate endgame,
    # whose class (seat 0 holds a known red) is not its own image, every meter says converged and
    # the value is right (-1/2), but the shipped profile is exploitable.
    game = _redblack("rb-pennies")
    if game is None:
        return
    s, meters = _run_mutant("color_swap_imposed", game)
    assert meters["converged"]
    assert abs(meters["self_value_p0"] + 0.5) < 1e-3
    assert _true_exploitability(s, game, tmp_path, "swap") > 1e-2
    s, meters = cfr.solve(game, "cfrplus", 4000, 250)
    assert meters["converged"]
    assert _true_exploitability(s, game, tmp_path, "clean") < 1e-3
