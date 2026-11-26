"""Tests for games/kuhn.py and games/leduc.py.

The heart of this file is a pair of small rules engines written straight from
the rules, sharing no code with the generators. Each engine plays a hand
(random or exhaustive), and the same moves are replayed on the generated tree:
at every decision node the tree's player, infoset id and legal action labels
must equal the engine's, and every terminal's payoff must equal the engine's.

Run: uv run pytest games/tests/test_poker.py -q
"""

from __future__ import annotations

import itertools
import json
import random
import sys
from fractions import Fraction
from pathlib import Path
from typing import ClassVar

import pytest

GAMES = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GAMES))

import common
import kuhn
import leduc

# ---------------------------------------------------------------------------
# Independent rules engines


class KuhnEngine:
    """Kuhn poker from the rules: ante 1, one bet of 1, player 0 first."""

    ORDER: ClassVar[dict[str, int]] = {"J": 0, "Q": 1, "K": 2}

    def __init__(self, c0: str, c1: str):
        self.cards = (c0, c1)
        self.put = [1, 1]
        self.moves: list[str] = []
        self.folded: int | None = None

    def to_act(self) -> int:
        return len(self.moves) % 2

    def bet_outstanding(self) -> bool:
        return self.put[0] != self.put[1]

    def done(self) -> bool:
        if self.folded is not None:
            return True
        m = "".join(self.moves)
        return m in ("cc", "bc", "cbc")

    def legal(self) -> list[str]:
        return ["fold", "call"] if self.bet_outstanding() else ["check", "bet"]

    def infoset(self) -> str:
        return f"{self.cards[self.to_act()]}:{''.join(self.moves)}"

    def play(self, action: str) -> None:
        p = self.to_act()
        assert action in self.legal()
        if action == "fold":
            self.folded = p
            self.moves.append("f")
        elif action == "bet":
            self.put[p] += 1
            self.moves.append("b")
        else:  # check or call: match the other player
            self.put[p] = self.put[1 - p]
            self.moves.append("c")

    def payoff_p0(self) -> Fraction:
        if self.folded is not None:
            return Fraction(-self.put[0] if self.folded == 0 else self.put[1])
        assert self.put[0] == self.put[1]
        w = 1 if self.ORDER[self.cards[0][0]] > self.ORDER[self.cards[1][0]] else -1
        return Fraction(w * self.put[0])


class LeducEngine:
    """Leduc hold'em from the rules (OpenSpiel leduc_poker defaults).

    Written with a pot / amount-to-call bookkeeping rather than the
    generator's stakes bookkeeping, so the two do not share a mistake.
    """

    RANK: ClassVar[dict[str, int]] = {"J": 1, "Q": 2, "K": 3}
    BET: ClassVar[dict[int, int]] = {1: 2, 2: 4}

    def __init__(self, c0: str, c1: str):
        self.private = (c0, c1)
        self.public: str | None = None
        self.pot = 2
        self.invested = [1, 1]
        self.round = 1
        self.hist = {1: "", 2: ""}
        self.to_call = 0  # what the player to act owes
        self.raises = 0
        self.player = 0
        self.folder: int | None = None
        self.need_public = False
        self.over = False

    def legal(self) -> list[str]:
        acts = []
        if self.to_call > 0:
            acts.append("fold")
        acts.append("call")
        if self.raises < 2:
            acts.append("raise")
        return acts

    def infoset(self) -> str:
        me = self.private[self.player]
        if self.round == 1:
            return f"{me}:{self.hist[1]}"
        return f"{me}|{self.public}:{self.hist[1]}/{self.hist[2]}"

    def play(self, action: str) -> None:
        assert not self.over and not self.need_public and action in self.legal()
        p = self.player
        if action == "fold":
            self.folder, self.over = p, True
            self.hist[self.round] += "f"
            return
        if action == "call":
            first = self.hist[self.round] == ""
            self.pot += self.to_call
            self.invested[p] += self.to_call
            self.to_call = 0
            self.hist[self.round] += "c"
            if not first:  # round over
                if self.round == 1:
                    self.need_public = True
                else:
                    self.over = True
                return
        else:  # raise: pay what is owed, then the round's bet
            pay = self.to_call + self.BET[self.round]
            self.pot += pay
            self.invested[p] += pay
            self.to_call = self.BET[self.round]
            self.raises += 1
            self.hist[self.round] += "r"
        self.player = 1 - p

    def deal_public(self, card: str) -> None:
        assert self.need_public and card not in self.private
        self.public = card
        self.need_public = False
        self.round, self.raises, self.player, self.to_call = 2, 0, 0, 0

    def strength(self, card: str) -> int:
        return 10 * (card[0] == self.public[0]) + self.RANK[card[0]]

    def payoff_p0(self) -> Fraction:
        assert self.over
        if self.folder is not None:
            winner = 1 - self.folder
        else:
            s0, s1 = self.strength(self.private[0]), self.strength(self.private[1])
            if s0 == s1:
                return Fraction(0)
            winner = 0 if s0 > s1 else 1
        # winner takes the pot; net to player 0 is pot share minus investment
        return Fraction(self.pot - self.invested[0] if winner == 0 else -self.invested[0])


# ---------------------------------------------------------------------------
# Fixtures


@pytest.fixture(scope="module")
def kuhn_game():
    return kuhn.build()


@pytest.fixture(scope="module")
def leduc_game():
    return leduc.build()


def by_label(items, label):
    hits = [x for x in items if x["label"] == label]
    assert len(hits) == 1, label
    return hits[0]["child"]


# ---------------------------------------------------------------------------
# Replay: engine moves on the tree


def replay_kuhn(game, deal: str, choose) -> tuple[Fraction, Fraction]:
    nodes = game["nodes"]
    eng = KuhnEngine(deal[0], deal[1])
    nid = by_label(nodes[game["root"]]["outcomes"], deal)
    while not eng.done():
        n = nodes[nid]
        assert n["kind"] == "decision"
        assert n["player"] == eng.to_act()
        assert n["infoset"] == eng.infoset()
        assert [a["label"] for a in n["actions"]] == eng.legal()
        a = choose(eng.legal())
        eng.play(a)
        nid = by_label(n["actions"], a)
    assert nodes[nid]["kind"] == "terminal"
    return eng.payoff_p0(), Fraction(nodes[nid]["payoff_p0"])


def replay_leduc(game, c0: str, c1: str, choose, choose_public) -> tuple[Fraction, Fraction]:
    nodes = game["nodes"]
    eng = LeducEngine(c0, c1)
    nid = by_label(nodes[game["root"]]["outcomes"], f"{c0},{c1}")
    while not eng.over:
        n = nodes[nid]
        if eng.need_public:
            assert n["kind"] == "chance"
            rest = [c for c in leduc.CARDS if c not in (c0, c1)]
            assert sorted(o["label"] for o in n["outcomes"]) == sorted(rest)
            assert all(Fraction(o["prob"]) == Fraction(1, 4) for o in n["outcomes"])
            card = choose_public(rest)
            eng.deal_public(card)
            nid = by_label(n["outcomes"], card)
            continue
        assert n["kind"] == "decision"
        assert n["player"] == eng.player
        assert n["infoset"] == eng.infoset()
        assert [a["label"] for a in n["actions"]] == eng.legal()
        a = choose(eng.legal())
        eng.play(a)
        nid = by_label(n["actions"], a)
    assert nodes[nid]["kind"] == "terminal"
    return eng.payoff_p0(), Fraction(nodes[nid]["payoff_p0"])


# ---------------------------------------------------------------------------
# Differential tests


def test_kuhn_random_playouts_match_tree(kuhn_game):
    rng = random.Random(2520)
    deals = ["".join(p) for p in itertools.permutations("JQK", 2)]
    for _ in range(5000):
        want, got = replay_kuhn(kuhn_game, rng.choice(deals), rng.choice)
        assert want == got


def test_kuhn_exhaustive_playouts_match_tree(kuhn_game):
    seen = 0
    for deal in ("".join(p) for p in itertools.permutations("JQK", 2)):
        for seq in itertools.product([0, 1], repeat=3):
            it = iter(seq)
            want, got = replay_kuhn(kuhn_game, deal, lambda legal, it=it: legal[next(it)])
            assert want == got
            seen += 1
    assert seen == 48


def test_kuhn_engine_hands_cover_every_terminal(kuhn_game):
    """Every Kuhn hand the engine allows lands on a distinct terminal, and together
    they cover all terminals (the tree has no hand the rules do not allow)."""
    nodes = kuhn_game["nodes"]
    reached: set[int] = set()

    def walk(deal: str, moves: tuple[str, ...]):
        eng = KuhnEngine(deal[0], deal[1])
        for m in moves:
            eng.play(m)
        if eng.done():
            nid = by_label(nodes[kuhn_game["root"]]["outcomes"], deal)
            for m in moves:
                nid = by_label(nodes[nid]["actions"], m)
            assert nodes[nid]["kind"] == "terminal" and nid not in reached
            assert Fraction(nodes[nid]["payoff_p0"]) == eng.payoff_p0()
            reached.add(nid)
            return
        for a in eng.legal():
            walk(deal, (*moves, a))

    for deal in ("".join(p) for p in itertools.permutations("JQK", 2)):
        walk(deal, ())
    assert reached == {n["id"] for n in nodes if n["kind"] == "terminal"}


def test_deal_probabilities_are_uniform(kuhn_game, leduc_game):
    k = kuhn_game["nodes"][kuhn_game["root"]]["outcomes"]
    assert sorted(o["label"] for o in k) == sorted("".join(p) for p in itertools.permutations("JQK", 2))
    assert all(o["prob"] == "1/6" for o in k)
    d = leduc_game["nodes"][leduc_game["root"]]["outcomes"]
    assert sorted(o["label"] for o in d) == sorted(f"{a},{b}" for a, b in itertools.permutations(leduc.CARDS, 2))
    assert all(o["prob"] == "1/30" for o in d)


def test_leduc_random_playouts_match_tree(leduc_game):
    rng = random.Random(2520)
    for _ in range(20000):
        c0, c1 = rng.sample(leduc.CARDS, 2)
        want, got = replay_leduc(leduc_game, c0, c1, rng.choice, rng.choice)
        assert want == got


def test_leduc_exhaustive_playouts_cover_every_terminal(leduc_game):
    """Enumerate every hand the engine allows; each must land on a distinct
    terminal with the engine's payoff, and together they must cover all
    terminals in the tree (so the tree has no hand the rules do not allow)."""
    nodes = leduc_game["nodes"]
    reached: set[int] = set()

    def walk(eng: LeducEngine, nid: int):
        n = nodes[nid]
        if eng.over:
            assert n["kind"] == "terminal"
            assert Fraction(n["payoff_p0"]) == eng.payoff_p0()
            assert nid not in reached
            reached.add(nid)
            return
        if eng.need_public:
            for card in leduc.CARDS:
                if card in eng.private:
                    continue
                e2 = clone(eng)
                e2.deal_public(card)
                walk(e2, by_label(n["outcomes"], card))
            return
        assert n["infoset"] == eng.infoset() and n["player"] == eng.player
        assert [a["label"] for a in n["actions"]] == eng.legal()
        for a in eng.legal():
            e2 = clone(eng)
            e2.play(a)
            walk(e2, by_label(n["actions"], a))

    def clone(eng: LeducEngine) -> LeducEngine:
        e = LeducEngine(*eng.private)
        e.__dict__.update({k: (dict(v) if isinstance(v, dict) else list(v) if isinstance(v, list) else v)
                           for k, v in eng.__dict__.items()})
        return e

    for c0, c1 in itertools.permutations(leduc.CARDS, 2):
        walk(LeducEngine(c0, c1), by_label(nodes[0]["outcomes"], f"{c0},{c1}"))
    terminals = {n["id"] for n in nodes if n["kind"] == "terminal"}
    assert reached == terminals
    assert len(terminals) == 5520


# ---------------------------------------------------------------------------
# Structure and counts


def test_kuhn_counts(kuhn_game):
    c = common.check_game(kuhn_game)
    assert c == {"nodes": 55, "chance": 1, "decision": 24, "terminal": 30,
                 "infosets": 12, "infosets_p0": 6, "infosets_p1": 6}


def test_leduc_counts(leduc_game):
    c = common.check_game(leduc_game)
    # 936 infosets: OpenSpiel leduc_poker. Node counts are derived in leduc.py's
    # docstring and by the exhaustive test; they are not a literature figure.
    assert c == {"nodes": 9451, "chance": 151, "decision": 3780, "terminal": 5520,
                 "infosets": 936, "infosets_p0": 468, "infosets_p1": 468}


def test_leduc_payoff_range(leduc_game):
    pays = {Fraction(n["payoff_p0"]) for n in leduc_game["nodes"] if n["kind"] == "terminal"}
    assert max(pays) == 13 and min(pays) == -13 and 0 in pays


def test_no_float_numbers():
    def hook(x):
        raise AssertionError(f"float {x}")

    for name in ("kuhn.game.json", "leduc.game.json", "kuhn.ref-strategy.json"):
        json.loads((GAMES / "out" / name).read_text(), parse_float=hook)


def test_files_on_disk_are_current(kuhn_game, leduc_game):
    assert json.loads((GAMES / "out" / "kuhn.game.json").read_text()) == kuhn_game
    assert json.loads((GAMES / "out" / "leduc.game.json").read_text()) == leduc_game
    assert json.loads((GAMES / "out" / "kuhn.ref-strategy.json").read_text()) == kuhn.ref_strategy()


def test_check_game_catches_broken_files(kuhn_game):
    import copy
    bad = copy.deepcopy(kuhn_game)
    bad["nodes"][0]["outcomes"][0]["prob"] = "1/5"
    with pytest.raises(AssertionError):
        common.check_game(bad)
    bad = copy.deepcopy(kuhn_game)
    # put one player 1 node into a player 0 infoset: the infoset now mixes players
    d = next(n for n in bad["nodes"] if n.get("infoset") == "J:c")
    d["infoset"] = "J:"
    with pytest.raises(AssertionError):
        common.check_game(bad)


# ---------------------------------------------------------------------------
# Kuhn reference strategy: exact value and exact exploitability by brute force


def kuhn_ev(game, s0, s1) -> Fraction:
    nodes = game["nodes"]

    def ev(nid):
        n = nodes[nid]
        if n["kind"] == "terminal":
            return Fraction(n["payoff_p0"])
        if n["kind"] == "chance":
            return sum(Fraction(o["prob"]) * ev(o["child"]) for o in n["outcomes"])
        s = s0 if n["player"] == 0 else s1
        return sum(s[n["infoset"]][a["label"]] * ev(a["child"]) for a in n["actions"])

    return ev(game["root"])


def pure_strategies(game, player):
    rows = {}
    for n in game["nodes"]:
        if n["kind"] == "decision" and n["player"] == player:
            rows[n["infoset"]] = [a["label"] for a in n["actions"]]
    keys = sorted(rows)
    for choice in itertools.product(*(rows[k] for k in keys)):
        yield {k: {a: Fraction(int(a == c)) for a in rows[k]} for k, c in zip(keys, choice)}


def exact_profile(strategy_doc):
    s = {k: {a: Fraction(p) for a, p in row.items()} for k, row in strategy_doc["probs"].items()}
    s0 = {k: v for k, v in s.items() if k.endswith((":", ":cb"))}
    s1 = {k: v for k, v in s.items() if k not in s0}
    return s0, s1


@pytest.mark.parametrize("alpha", [Fraction(0), Fraction(1, 6), Fraction(1, 3)])
def test_kuhn_equilibrium_family_is_exact(kuhn_game, alpha):
    doc = kuhn.ref_strategy(alpha)
    for row in doc["probs"].values():
        assert sum(Fraction(p) for p in row.values()) == 1
    s0, s1 = exact_profile(doc)
    assert len(s0) == 6 and len(s1) == 6
    assert kuhn_ev(kuhn_game, s0, s1) == Fraction(-1, 18)
    br0 = max(kuhn_ev(kuhn_game, p, s1) for p in pure_strategies(kuhn_game, 0))
    br1 = max(-kuhn_ev(kuhn_game, s0, p) for p in pure_strategies(kuhn_game, 1))
    assert br0 == Fraction(-1, 18) and br1 == Fraction(1, 18)
    assert br0 + br1 == 0  # NashConv, hence exploitability, exactly 0


def test_kuhn_perturbed_strategy_is_exploitable(kuhn_game):
    """Sanity: the brute force is not vacuous. Player 1 calling with J is a mistake."""
    doc = kuhn.ref_strategy()
    doc["probs"]["J:b"] = {"fold": "0", "call": "1"}
    s0, s1 = exact_profile(doc)
    br0 = max(kuhn_ev(kuhn_game, p, s1) for p in pure_strategies(kuhn_game, 0))
    br1 = max(-kuhn_ev(kuhn_game, s0, p) for p in pure_strategies(kuhn_game, 1))
    assert br0 + br1 > 0


# ---------------------------------------------------------------------------
# Mutation check: the differential test must fail on a wrong generator


def _no_pair_rule(c0: str, c1: str, pub: str) -> int:
    r0, r1 = leduc.RANKS.index(c0[0]), leduc.RANKS.index(c1[0])
    return (r0 > r1) - (r0 < r1)


@pytest.mark.parametrize("mutation", ["round2_raise_2", "no_pair_rule", "three_raises", "round1_raise_4"])
def test_leduc_differential_catches_rule_mutations(monkeypatch, mutation):
    if mutation == "round2_raise_2":
        monkeypatch.setitem(leduc.RAISE_SIZE, 2, 2)
    elif mutation == "round1_raise_4":
        monkeypatch.setitem(leduc.RAISE_SIZE, 1, 4)
    elif mutation == "no_pair_rule":
        monkeypatch.setattr(leduc, "showdown_sign", _no_pair_rule)
    else:
        monkeypatch.setattr(leduc, "MAX_RAISES", 3)
    bad = leduc.build()
    rng = random.Random(2520)
    with pytest.raises(AssertionError):
        for _ in range(20000):
            c0, c1 = rng.sample(leduc.CARDS, 2)
            want, got = replay_leduc(bad, c0, c1, rng.choice, rng.choice)
            assert want == got
