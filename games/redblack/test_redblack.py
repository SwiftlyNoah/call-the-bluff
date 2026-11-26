"""Pins the Red/Black mini results. Run: uv run pytest games/redblack -q

Needs the files from `uv run games/redblack/generate.py` and the engine mirror
in ~/coding/red_black/simulator (read only).
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import difftest as D  # noqa: E402
import generate as G  # noqa: E402
import model as M  # noqa: E402
import solve as S  # noqa: E402

OUT = HERE.parent / "out" / "redblack"


@pytest.mark.parametrize("name", ["rb-h1", "rb-pennies-inner", "rb-pennies", "rb-21", "rb-12"])
def test_file_matches_engine_exhaustively(name: str) -> None:
    doc = json.loads((OUT / f"{name}.game.json").read_text())
    rep = D.walk(doc, G.VARIANTS[name], "all", 0)
    assert rep.nodes >= len(doc["nodes"])
    assert not rep.errors, rep.errors[:3]
    assert not rep.split, rep.split[:3]
    assert not rep.merged, rep.merged[:3]


def test_file_is_what_the_generator_writes() -> None:
    for name in G.DEFAULT:
        doc = json.loads((OUT / f"{name}.game.json").read_text())
        assert doc["nodes"] == G.build(G.VARIANTS[name]), name


def test_leaky_mutant_is_caught_as_a_leak() -> None:
    doc = json.loads((OUT / "rb-pennies-leaky.game.json").read_text())
    rep = D.walk(doc, G.VARIANTS["rb-pennies-leaky"], "all", 0)
    assert rep.split and not rep.errors


@pytest.mark.parametrize("mutant", [m for m in M.MUTANTS if m != "reorder-round-1-only"])
def test_seeded_spec_bugs_are_caught(mutant: str) -> None:
    cfg = replace(G.VARIANTS["rb-21"], mutant=mutant)
    rep = D.walk({"root": 0, "nodes": G.build(cfg)}, cfg, "first", 0)
    assert rep.errors or rep.split or rep.merged


def test_round_one_only_reorder_is_equivalent_on_rb21() -> None:
    # After round 1 of rb-21 no hand has two cards, so there is no later
    # reorder to drop: an equivalent mutant, which the test must NOT flag.
    cfg = replace(G.VARIANTS["rb-21"], mutant="reorder-round-1-only")
    rep = D.walk({"root": 0, "nodes": G.build(cfg)}, cfg, "first", 0)
    assert not (rep.errors or rep.split or rep.merged)


def _certify(name: str, iters: int = 1500) -> Fraction:
    g = S.load(OUT / f"{name}.game.json")
    avg = S.cfr_plus(g, iters)
    for d in range(1, 65):
        sigma = S.rounded(g, avg, d)
        if S.best_response(g, sigma, 0)[0] + S.best_response(g, sigma, 1)[0] == 0:
            return S.profile_value(g, sigma)
    raise AssertionError("no exact certificate")


def test_pennies_is_exactly_25_percent() -> None:
    v = _certify("rb-pennies")
    assert v == Fraction(-1, 2)  # P(seat 0 wins) = (1 + v) / 2 = 1/4


def test_leak_moves_the_value_to_50_percent() -> None:
    assert _certify("rb-pennies-leaky") == 0


def test_inner_pennies_is_even() -> None:
    assert _certify("rb-pennies-inner") == 0


def test_h1_closed_form() -> None:
    # One card each: the opener wins exactly when the colors differ.
    p_diff = Fraction(2 * 26 * 26, 52 * 51)
    assert _certify("rb-h1") == p_diff - (1 - p_diff) == Fraction(1, 51)


# --- Added by review-redblack-mini: the test's own blind spots ------------


def test_leaky_file_leaks_every_extra_infoset() -> None:
    # The leak count is the number of DISTINCT leaked infoset pairs, not the
    # 20-example print cap: the leaky file has 317 seat-0 infosets where the
    # engine allows 167, and every one of the 150 extra splits is reported.
    doc = json.loads((OUT / "rb-pennies-leaky.game.json").read_text())
    rep = D.walk(doc, G.VARIANTS["rb-pennies-leaky"], "all", 0)
    assert rep.n_split == 317 - 167 == 150


def test_an_engine_that_ships_stable_card_ids_is_caught(monkeypatch) -> None:
    # Lesson 1 from the engine side: if the projection gave the OPPONENT's
    # cards their real ids (as the v0 solver's stable ids did), seat 0 could
    # tell which card seat 1 discarded. The file (correctly) cannot, so the
    # walk must report merges. An earlier canonicalization mapped every id
    # starting with "c" to "own" and passed this broken engine silently.
    orig = D.project_for_recipient

    def leaky(state, uid):
        v = orig(state, uid)
        seats = tuple(
            replace(s, hand=tuple(replace(c, id=real.id) for c, real in zip(s.hand, state.seats[s.seat].hand)))
            for s in v.seats
        )
        return replace(v, seats=seats)

    monkeypatch.setattr(D, "project_for_recipient", leaky)
    doc = json.loads((OUT / "rb-pennies.game.json").read_text())
    rep = D.walk(doc, G.VARIANTS["rb-pennies"], "all", 0)
    assert rep.n_merged > 0


def test_a_wrong_deal_probability_is_caught() -> None:
    doc = json.loads((OUT / "rb-h1.game.json").read_text())
    outs = doc["nodes"][doc["root"]]["outcomes"]
    a, b = Fraction(outs[0]["prob"]), Fraction(outs[1]["prob"])
    outs[0]["prob"], outs[1]["prob"] = str(a + Fraction(1, 1000)), str(b - Fraction(1, 1000))
    rep = D.walk(doc, G.VARIANTS["rb-h1"], "first", 0)
    assert any(e.startswith("deal:") for e in rep.errors)


def test_open_discard_mutant_is_a_leak_on_the_gate() -> None:
    cfg = replace(G.VARIANTS["rb-pennies"], mutant="open-discard")
    rep = D.walk({"root": 0, "nodes": G.build(cfg)}, cfg, "all", 0)
    assert rep.n_split > 0 and not rep.errors
