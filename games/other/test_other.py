"""Tests for games/other/. Run from the repo root:  uv run pytest games/other -q"""

from __future__ import annotations

import copy
import itertools
import json
import random
import sys
from fractions import Fraction
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_all
import matrix_games
from differential import Checker, Mismatch
from gamelib import (
    OUT_DIR,
    Tree,
    best_response,
    expected_value,
    exploitability,
    game_doc,
    infoset_actions,
    load_game,
    lp_bracket,
    validate,
)
from referees import REFEREES

GAMES = {doc["name"]: doc for doc in build_all.all_games()}
NAMES = sorted(GAMES)


def test_registry_complete():
    assert set(GAMES) == set(REFEREES)


@pytest.mark.parametrize("name", NAMES)
def test_written_file_matches_generator(name):
    """games/out/other is up to date with the generators (rerun build_all.py if this fails)."""
    path = OUT_DIR / f"{name}.game.json"
    assert load_game(path) == json.loads(json.dumps(GAMES[name]))


@pytest.mark.parametrize("name", NAMES)
def test_valid(name):
    validate(GAMES[name])


@pytest.mark.parametrize("name", NAMES)
def test_differential_exhaustive_and_random(name):
    c = Checker(GAMES[name], REFEREES[name])
    assert c.exhaustive() == len(GAMES[name]["nodes"])
    c.random_playouts(500, random.Random(7))
    assert len(c.iset_to_obs) == len(infoset_actions(GAMES[name]))


# ---- the differential test must catch real game-file bugs (mutation check)


def _mutants(doc):
    nodes = doc["nodes"]
    terms = [n for n in nodes if n["kind"] == "terminal" and Fraction(n["payoff_p0"]) != 0]
    decs = [n for n in nodes if n["kind"] == "decision" and len(n["actions"]) >= 2]
    chances = [n for n in nodes if n["kind"] == "chance" and len(n["outcomes"]) >= 2]
    out = {}

    def mut(tag, f):
        m = copy.deepcopy(doc)
        f({n["id"]: n for n in m["nodes"]})
        out[tag] = m

    t = terms[len(terms) // 2]
    mut("payoff_sign", lambda b: b[t["id"]].update(payoff_p0=str(-Fraction(t["payoff_p0"]))))
    d = decs[len(decs) // 2]

    def swap(b):
        acts = b[d["id"]]["actions"]
        acts[0]["child"], acts[1]["child"] = acts[1]["child"], acts[0]["child"]

    mut("swap_children", swap)
    # a real split needs an infoset with at least two nodes (renaming a singleton changes nothing)
    sizes = {}
    for n in decs:
        sizes[n["infoset"]] = sizes.get(n["infoset"], 0) + 1
    s = next(n for n in decs if sizes[n["infoset"]] >= 2)
    mut("split_infoset", lambda b: b[s["id"]].update(infoset=s["infoset"] + "#split"))
    p1 = [n for n in decs if n["player"] == 1]
    other = next(
        (
            n
            for n in p1
            if n["infoset"] != p1[0]["infoset"]
            and [a["label"] for a in n["actions"]] == [a["label"] for a in p1[0]["actions"]]
        ),
        None,
    )
    if other is not None:
        mut("merge_infosets", lambda b: b[other["id"]].update(infoset=p1[0]["infoset"]))
    if chances:
        c = chances[0]

        def shift(b):
            o = b[c["id"]]["outcomes"]
            eps = min(Fraction(o[0]["prob"]), Fraction(o[1]["prob"])) / 2
            o[0]["prob"] = str(Fraction(o[0]["prob"]) + eps)
            o[1]["prob"] = str(Fraction(o[1]["prob"]) - eps)

        mut("chance_prob", shift)
    return out


@pytest.mark.parametrize("name", NAMES)
def test_differential_catches_mutants(name):
    mutants = _mutants(GAMES[name])
    assert mutants
    for m in mutants.values():
        with pytest.raises(Mismatch):
            Checker(m, REFEREES[name]).exhaustive()


# ---- values


@pytest.mark.parametrize("name", NAMES)
def test_known_value_certified_by_lp_profile(name):
    doc = GAMES[name]
    b = lp_bracket(doc)
    assert b["lower"] == b["upper"], (b["lower"], b["upper"])
    assert b["lower"] == Fraction(doc["known_value_p0"])
    e = exploitability(doc, b["profile"])
    assert e["nash_conv"] == 0
    assert expected_value(doc, b["profile"]) == b["lower"]


def test_rps_biased_hand_equilibrium():
    doc = GAMES["rps_biased"]
    eq = matrix_games.RPS_BIASED_EQ
    assert exploitability(doc, eq)["nash_conv"] == 0
    assert expected_value(doc, eq) == Fraction(1, 12)
    # and the unbiased RPS equilibrium is NOT one here
    uni = {iset: {a: Fraction(1, 3) for a in "RPS"} for iset in ("P0", "P1")}
    assert exploitability(doc, uni)["nash_conv"] > 0


def _brute_force_br(doc, profile, p):
    acts = {iset: labels for iset, (q, labels) in infoset_actions(doc).items() if q == p}
    keys = sorted(acts)
    best = None
    for choice in itertools.product(*(acts[iset] for iset in keys)):
        pure = {iset: {a: Fraction(int(a == c)) for a in acts[iset]} for iset, c in zip(keys, choice)}
        v = expected_value(doc, {**profile, **pure})
        v = v if p == 0 else -v
        best = v if best is None else max(best, v)
    return best


@pytest.mark.parametrize("name", ["rps_biased", "goofspiel3_fixed", "matching_pennies"])
def test_best_response_matches_brute_force(name):
    doc = GAMES[name]
    rng = random.Random(11)
    for _ in range(3):
        prof = {}
        for iset, (_, labels) in infoset_actions(doc).items():
            w = [Fraction(rng.randint(0, 5)) for _ in labels]
            if sum(w) == 0:
                w[0] = Fraction(1)
            prof[iset] = {a: x / sum(w) for a, x in zip(labels, w)}
        for p in (0, 1):
            assert best_response(doc, prof, p)[0] == _brute_force_br(doc, prof, p)


def test_lp_pipeline_on_kuhn():
    """Sanity check of the local LP + exact BR on Kuhn poker, whose value -1/18 is classical.
    The Kuhn tree here is a throwaway built inline; the project's Kuhn file is games/kuhn.py's."""
    t = Tree()
    cards = "JQK"

    def node(c, hist):
        ends = {"cc": None, "bc": "call", "bf": "fold", "cbc": "call", "cbf": "fold"}
        if hist in ends:
            p0_wins = cards.index(c[0]) > cards.index(c[1])
            if hist == "cc":
                return t.terminal(1 if p0_wins else -1)
            if ends[hist] == "call":
                return t.terminal(2 if p0_wins else -2)
            return t.terminal(1 if hist == "bf" else -1)  # the folder loses the ante
        player = len(hist) % 2
        labels = ["c", "b"] if hist in ("", "c") else ["f", "c"]
        return t.decision(
            player,
            f"{c[player]}:{hist}",
            [(a, lambda a=a: node(c, hist + a)) for a in labels],
        )

    deals = [(a, b) for a in cards for b in cards if a != b]
    t.chance([(a + b, Fraction(1, 6), lambda a=a, b=b: node((a, b), "")) for a, b in deals])
    doc = game_doc("kuhn_inline", "Kuhn poker", t)
    validate(doc)
    b = lp_bracket(doc)
    assert b["lower"] == b["upper"] == Fraction(-1, 18)
