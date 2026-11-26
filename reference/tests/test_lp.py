"""Tests for reference/lp.py, using game builders local to this file so they do not depend
on games/ (written concurrently by other agents). If games/out/kuhn.game.json exists, it is
checked too."""

from __future__ import annotations

import itertools
import json
import random
import sys
from fractions import Fraction
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "reference"))

import lp  # noqa: E402


class Builder:
    def __init__(self):
        self.nodes = []

    def add(self, node):
        node["id"] = len(self.nodes)
        self.nodes.append(node)
        return node["id"]

    def game(self, name, root):
        return {"format": "bluff-game-v1", "name": name, "root": root, "nodes": self.nodes}


def kuhn_dict():
    """3-card Kuhn poker, ante 1, bet 1. Player 0 acts first."""
    b = Builder()
    cards = "JQK"

    def term(v):
        return b.add({"kind": "terminal", "payoff_p0": str(v)})

    def dec(player, infoset, acts):
        nid = b.add({"kind": "decision", "player": player, "infoset": infoset, "actions": []})
        b.nodes[nid]["actions"] = [{"label": a, "child": c()} for a, c in acts]
        return nid

    root = b.add({"kind": "chance", "outcomes": []})
    outs = []
    for c0, c1 in itertools.permutations(range(3), 2):
        win = 1 if c0 > c1 else -1
        h0, h1 = cards[c0], cards[c1]
        node = dec(0, f"{h0}:", [
            ("check", lambda: dec(1, f"{h1}:c", [
                ("check", lambda: term(win)),
                ("bet", lambda: dec(0, f"{h0}:cb", [
                    ("fold", lambda: term(-1)),
                    ("call", lambda: term(2 * win)),
                ])),
            ])),
            ("bet", lambda: dec(1, f"{h1}:b", [
                ("fold", lambda: term(1)),
                ("call", lambda: term(2 * win)),
            ])),
        ])
        outs.append({"label": h0 + h1, "prob": "1/6", "child": node})
    b.nodes[root]["outcomes"] = outs
    return b.game("kuhn", root)


def matrix_game_dict(M, name="matrix"):
    """Simultaneous matrix game in tree form: p1 does not see p0's move."""
    b = Builder()
    root = b.add({"kind": "decision", "player": 0, "infoset": "p0", "actions": []})
    acts = []
    for i, row in enumerate(M):
        n1 = b.add({"kind": "decision", "player": 1, "infoset": "p1", "actions": []})
        b.nodes[n1]["actions"] = [{"label": f"c{j}", "child": b.add({"kind": "terminal", "payoff_p0": str(v)})}
                                  for j, v in enumerate(row)]
        acts.append({"label": f"r{i}", "child": n1})
    b.nodes[root]["actions"] = acts
    return b.game(name, root)


def solve(d):
    g = lp.game_from_dict(json.loads(json.dumps(d)))
    r = lp.solve_game(g)
    cert = lp.certify_profile(g, r.beh)
    return g, r, cert


def test_kuhn_value_exact():
    g, r, cert = solve(kuhn_dict())
    assert r.certified
    assert r.value == Fraction(-1, 18)
    assert cert["exploitability"] == 0
    assert cert["value_p0"] == Fraction(-1, 18)


def test_kuhn_file_if_present():
    path = ROOT / "games" / "out" / "kuhn.game.json"
    if not path.exists():
        pytest.skip("games/out/kuhn.game.json not written yet")
    g = lp.load_game(path)
    r = lp.solve_game(g)
    assert r.certified and r.value == Fraction(-1, 18)
    assert lp.certify_profile(g, r.beh)["exploitability"] == 0


def test_matching_pennies():
    _, r, cert = solve(matrix_game_dict([[1, -1], [-1, 1]]))
    assert r.value == 0 and cert["exploitability"] == 0
    assert r.beh["p0"] == {"r0": Fraction(1, 2), "r1": Fraction(1, 2)}


def test_rps_biased():
    # RPS with a doubled payoff for rock beating scissors: value 0, exact (1/4, 1/2, 1/4)-type mix.
    M = [[0, -1, 2], [1, 0, -1], [-2, 1, 0]]
    _, r, cert = solve(matrix_game_dict(M))
    assert r.certified and r.value == 0 and cert["exploitability"] == 0
    assert r.beh["p0"] == {"r0": Fraction(1, 4), "r1": Fraction(1, 2), "r2": Fraction(1, 4)}


def test_awkward_denominators():
    # A 2x2 game whose value has a denominator rounding at 1e3 would miss.
    M = [[Fraction(3, 7), Fraction(-5, 11)], [Fraction(-2, 13), Fraction(9, 17)]]
    _, r, cert = solve(matrix_game_dict([[str(v) for v in row] for row in M]))
    a, b_, c, d = M[0][0], M[0][1], M[1][0], M[1][1]
    expected = (a * d - b_ * c) / (a + d - b_ - c)
    assert r.certified and r.value == expected and cert["exploitability"] == 0


@pytest.mark.parametrize("seed", range(8))
def test_random_matrix_games(seed):
    rng = random.Random(seed)
    n, m = rng.randint(2, 5), rng.randint(2, 5)
    M = [[Fraction(rng.randint(-20, 20), rng.randint(1, 9)) for _ in range(m)] for _ in range(n)]
    _, r, cert = solve(matrix_game_dict([[str(v) for v in row] for row in M]))
    assert r.certified and cert["exploitability"] == 0 and cert["value_p0"] == r.value


def test_exact_br_detects_exploitable_profile():
    g = lp.game_from_dict(kuhn_dict())
    uniform = {iset: {a: Fraction(1, len(info["actions"])) for a in info["actions"]}
               for iset, info in g.infosets.items()}
    cert = lp.certify_profile(g, uniform)
    # Uniform play in Kuhn: value 1/8, BR values 1/2 and 5/12, nash_conv 11/12. The independent
    # reference/br.py must agree exactly.
    assert cert["value_p0"] == Fraction(1, 8)
    assert (cert["br_value_p0"], cert["br_value_p1"]) == (Fraction(1, 2), Fraction(5, 12))
    assert cert["nash_conv"] == Fraction(11, 12)
    br, vg = _reference_checker()
    v = br.compute_verdict(vg.validate_game_doc(kuhn_dict()), uniform)
    assert Fraction(v["nash_conv"]) == Fraction(11, 12) and Fraction(v["value_p0"]) == Fraction(1, 8)


def test_strategy_json_format():
    d = kuhn_dict()
    g = lp.game_from_dict(d)
    r = lp.solve_game(g)
    out = lp.strategy_json(g, r)
    assert out["format"] == "bluff-strategy-v1" and out["game"] == "kuhn"
    assert set(out["probs"]) == set(g.infosets)
    for iset, row in out["probs"].items():
        assert sum(Fraction(v) for v in row.values()) == 1
        assert list(row) == g.infosets[iset]["actions"]


def test_perfect_recall_violation_rejected():
    # p0 forgets its own first move: two nodes of infoset "x" have different parent sequences.
    b = Builder()
    root = b.add({"kind": "decision", "player": 0, "infoset": "a", "actions": []})
    kids = []
    for lab in ("l", "r"):
        n = b.add({"kind": "decision", "player": 0, "infoset": "x", "actions": [
            {"label": "u", "child": b.add({"kind": "terminal", "payoff_p0": "1"})},
            {"label": "d", "child": b.add({"kind": "terminal", "payoff_p0": "0"})}]})
        kids.append({"label": lab, "child": n})
    b.nodes[root]["actions"] = kids
    with pytest.raises(ValueError, match="perfect recall"):
        lp.solve_game(lp.game_from_dict(b.game("forgetful", root)))


def _pure_strategies(g, p):
    infos = sorted(iset for iset, info in g.infosets.items() if info["player"] == p)
    for combo in itertools.product(*(g.infosets[iset]["actions"] for iset in infos)):
        yield {iset: {a: Fraction(int(a == pick)) for a in g.infosets[iset]["actions"]}
               for iset, pick in zip(infos, combo)}


@pytest.mark.parametrize("p", [0, 1])
def test_best_response_matches_brute_force_on_kuhn(p):
    """The exact BR equals the max over every pure strategy (2^6 per player in Kuhn),
    against a fixed lopsided behavioral strategy of the opponent."""
    g = lp.game_from_dict(kuhn_dict())
    rng = random.Random(7 + p)
    opp = {}
    for iset, info in g.infosets.items():
        if info["player"] != p:
            w = [Fraction(rng.randint(1, 9)) for _ in info["actions"]]
            opp[iset] = {a: wi / sum(w) for a, wi in zip(info["actions"], w)}
    sign = 1 if p == 0 else -1
    brute = max(sign * lp.profile_value(g, {**opp, **pure}) for pure in _pure_strategies(g, p))
    val, choice = lp.best_response_value(g, opp, p)
    assert val == brute
    pure = {iset: {a: Fraction(int(a == choice[iset])) for a in g.infosets[iset]["actions"]} for iset in choice}
    assert sign * lp.profile_value(g, {**opp, **pure}) == val


def test_exact_vertex_path_alone(monkeypatch):
    """With rounding disabled, the exact vertex reconstruction must certify on its own."""
    monkeypatch.setattr(lp, "DENOMS", [])
    for d in (kuhn_dict(), matrix_game_dict([["3/7", "-5/11"], ["-2/13", "9/17"]])):
        g, r, cert = solve(d)
        assert r.method == {"p0": "exact vertex", "p1": "exact vertex"}
        assert r.certified and cert["exploitability"] == 0
    assert r.value is not None


def _reference_checker():
    """reference/br.py and reference/validate_game.py, used only as an outside cross-check."""
    br = pytest.importorskip("br")
    vg = pytest.importorskip("validate_game")
    return br, vg


@pytest.mark.parametrize("bad", ["1e3", " 1/2", "1_0", "0x1", "1/0", "nan"])
def test_rejects_non_contract_numbers(bad):
    d = matrix_game_dict([["1", bad], ["0", "1"]])
    with pytest.raises(ValueError):
        lp.game_from_dict(d)


def test_rejects_duplicate_action_labels():
    d = matrix_game_dict([["1", "0"], ["0", "1"]])
    d["nodes"][0]["actions"][1]["label"] = "r0"
    with pytest.raises(ValueError, match="distinct"):
        lp.game_from_dict(d)


def random_game_dict(seed, max_depth=6):
    """A random two-player zero-sum game with chance, perfect recall and nontrivial infosets.

    An infoset is the acting player's own observation history: their own past actions, the
    chance outcomes they were shown and the opponent actions they saw. Perfect recall holds
    by construction; the number of actions is a function of the infoset."""
    rng = random.Random(seed)
    b = Builder()

    def gen(depth, obs):
        r = rng.random()
        if depth >= max_depth or r < 0.15:
            v = Fraction(rng.randint(-9, 9), rng.randint(1, 4))
            return b.add({"kind": "terminal", "payoff_p0": lp.frac_str(v)})
        if r < 0.35:
            w = [rng.randint(1, 5) for _ in range(rng.randint(2, 3))]
            seen = rng.randint(0, 3)  # bit p set: player p sees the outcome
            nid = b.add({"kind": "chance", "outcomes": []})
            outs = []
            for i, wi in enumerate(w):
                o = tuple(obs[p] + ((f"c{i}",) if seen >> p & 1 else ()) for p in (0, 1))
                outs.append({"label": f"o{i}", "prob": lp.frac_str(Fraction(wi, sum(w))),
                             "child": gen(depth + 1, o)})
            b.nodes[nid]["outcomes"] = outs
            return nid
        p = rng.randint(0, 1)
        iset = f"P{p}|" + ".".join(obs[p])
        n_actions = random.Random(iset).randint(1, 3)
        opp_sees = rng.random() < 0.5
        nid = b.add({"kind": "decision", "player": p, "infoset": iset, "actions": []})
        acts = []
        for a in range(n_actions):
            o = [obs[0], obs[1]]
            o[p] = o[p] + (f"{iset}>a{a}",)
            if opp_sees:
                o[1 - p] = o[1 - p] + (f"saw{a}",)
            acts.append({"label": f"a{a}", "child": gen(depth + 1, tuple(o))})
        b.nodes[nid]["actions"] = acts
        return nid

    root = gen(0, ((), ()))
    return b.game(f"random{seed}", root)


@pytest.mark.parametrize("seed", range(40))
def test_random_extensive_games_cross_checked(seed):
    """Random perfect-recall games with chance and multi-node infosets: lp.py certifies, and the
    independent reference/br.py (which brute-forces small games itself) agrees that the profile
    has nash_conv 0 and the same value, and agrees on BR values against a random profile."""
    br, vg = _reference_checker()
    d = random_game_dict(seed)
    G = vg.validate_game_doc(json.loads(json.dumps(d)))
    g, r, cert = solve(d)
    if not g.infosets:
        pytest.skip("degenerate draw: no decisions")
    assert r.certified and cert["exploitability"] == 0 and cert["value_p0"] == r.value
    v = br.compute_verdict(G, r.beh)
    assert Fraction(v["nash_conv"]) == 0 and Fraction(v["value_p0"]) == r.value
    rng = random.Random(1000 + seed)
    sigma = {}
    for iset, info in g.infosets.items():
        w = [Fraction(rng.randint(0, 5)) for _ in info["actions"]]
        w[0] += sum(w) == 0
        sigma[iset] = {a: wi / sum(w) for a, wi in zip(info["actions"], w)}
    v = br.compute_verdict(G, sigma)
    assert lp.best_response_value(g, sigma, 0)[0] == Fraction(v["br_value_p0"])
    assert lp.best_response_value(g, sigma, 1)[0] == Fraction(v["br_value_p1"])


def test_random_games_have_real_infosets():
    """Guard that the fuzz above exercises imperfect information, not just perfect-info trees."""
    multi = 0
    for seed in range(40):
        g = lp.game_from_dict(random_game_dict(seed))
        multi += any(len(info["nodes"]) > 1 for info in g.infosets.values())
    assert multi >= 25
