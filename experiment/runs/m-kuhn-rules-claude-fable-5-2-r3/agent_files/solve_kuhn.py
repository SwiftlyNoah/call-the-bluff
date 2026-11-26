#!/usr/bin/env python3
"""Solve Kuhn poker with CFR+ and write the average strategy to strategy.json.

The game is implemented here from its rules. The solver uses CFR+ (regret
matching+, alternating updates, linearly weighted averaging). After solving,
the strategy file is read back and its exploitability and value are computed
exactly with rational arithmetic, by two independent best-response methods.
"""

import itertools
import json
import sys
from decimal import Decimal
from fractions import Fraction

CARDS = "JQK"
RANK = {"J": 0, "Q": 1, "K": 2}
DEALS = [(a, b) for a in CARDS for b in CARDS if a != b]  # (player 0 card, player 1 card)

# ---------------------------------------------------------------------------
# Game definition
#
# A history is a tuple of action labels. Information set names use the letters
# "c" (check) and "b" (bet) for the actions taken so far; fold and call always
# end the hand, so they never appear in an information set name.
# ---------------------------------------------------------------------------

LETTER = {"check": "c", "bet": "b"}


def is_terminal(history):
    if not history:
        return False
    last = history[-1]
    return last in ("fold", "call") or history == ("check", "check")


def player_to_act(history):
    return len(history) % 2


def legal_actions(history):
    if history and history[-1] == "bet":
        return ("fold", "call")
    return ("check", "bet")


def payoff_p0(history, deal):
    """Net payoff to player 0 at a terminal history."""
    last = history[-1]
    if last == "fold":
        folder = (len(history) - 1) % 2
        return -1 if folder == 0 else 1
    stake = 2 if last == "call" else 1  # ante, plus the matched bet after a call
    return stake if RANK[deal[0]] > RANK[deal[1]] else -stake


def infoset_name(history, deal):
    card = deal[player_to_act(history)]
    return card + ":" + "".join(LETTER[a] for a in history)


def all_decision_histories():
    out = []

    def walk(history):
        if is_terminal(history):
            return
        out.append(history)
        for a in legal_actions(history):
            walk(history + (a,))

    walk(())
    return out


DECISION_HISTORIES = all_decision_histories()
INFOSETS = {}  # name -> (player, actions)
for _h in DECISION_HISTORIES:
    for _deal in DEALS:
        INFOSETS[infoset_name(_h, _deal)] = (player_to_act(_h), legal_actions(_h))

# ---------------------------------------------------------------------------
# CFR+
# ---------------------------------------------------------------------------


class CFRPlus:
    def __init__(self):
        self.regret = {k: [0.0] * len(v[1]) for k, v in INFOSETS.items()}
        self.strategy_sum = {k: [0.0] * len(v[1]) for k, v in INFOSETS.items()}

    def current_strategy(self, name):
        r = self.regret[name]
        total = sum(r)
        if total > 0:
            return [x / total for x in r]
        return [1.0 / len(r)] * len(r)

    def _walk(self, history, deal, reach, updating, weight, delta):
        """Returns the expected payoff to the updating player.

        reach[i] is player i's own contribution to the reach probability;
        the chance probability is folded into the opponent's entry.
        """
        if is_terminal(history):
            v = payoff_p0(history, deal)
            return v if updating == 0 else -v
        player = player_to_act(history)
        actions = legal_actions(history)
        name = infoset_name(history, deal)
        sigma = self.current_strategy(name)
        values = []
        for i, a in enumerate(actions):
            nxt = list(reach)
            nxt[player] *= sigma[i]
            values.append(self._walk(history + (a,), deal, nxt, updating, weight, delta))
        node_value = sum(s * v for s, v in zip(sigma, values))
        if player == updating:
            opp = reach[1 - player]
            d = delta.setdefault(name, [0.0] * len(actions))
            ssum = self.strategy_sum[name]
            for i in range(len(actions)):
                d[i] += opp * (values[i] - node_value)
                ssum[i] += weight * reach[player] * sigma[i]
        return node_value

    def iterate(self, t):
        for updating in (0, 1):
            delta = {}
            for deal in DEALS:
                reach = [1.0, 1.0]
                reach[1 - updating] = 1.0 / len(DEALS)
                self._walk((), deal, reach, updating, float(t), delta)
            # Regret matching+: add the whole iteration's regret, then floor at zero.
            for name, d in delta.items():
                r = self.regret[name]
                for i in range(len(r)):
                    r[i] = max(r[i] + d[i], 0.0)

    def average_strategy(self):
        out = {}
        for name, s in self.strategy_sum.items():
            total = sum(s)
            out[name] = [x / total for x in s] if total > 0 else [1.0 / len(s)] * len(s)
        return out


# ---------------------------------------------------------------------------
# Exact evaluation (rational arithmetic)
# ---------------------------------------------------------------------------


def expected_value_p0(probs):
    """Expected payoff to player 0 when both players follow probs."""

    def walk(history, deal):
        if is_terminal(history):
            return Fraction(payoff_p0(history, deal))
        name = infoset_name(history, deal)
        return sum(probs[name][a] * walk(history + (a,), deal) for a in legal_actions(history))

    return sum(walk((), deal) for deal in DEALS) / len(DEALS)


def best_response_value(probs, br_player):
    """Best-response value for br_player against the other player's part of probs.

    Computed by backward induction over br_player's information sets: at each
    one, the action maximizing the counterfactual value summed over all the
    histories in that information set is chosen.
    """
    chosen = {}

    def cf_value(history, deal, opp_reach):
        if is_terminal(history):
            v = payoff_p0(history, deal)
            return opp_reach * (v if br_player == 0 else -v)
        name = infoset_name(history, deal)
        if player_to_act(history) == br_player:
            return cf_value(history + (chosen[name],), deal, opp_reach)
        return sum(
            cf_value(history + (a,), deal, opp_reach * probs[name][a])
            for a in legal_actions(history)
        )

    def opp_reach_to(history, deal):
        reach = Fraction(1)
        for i in range(len(history)):
            prefix = history[:i]
            if player_to_act(prefix) != br_player:
                reach *= probs[infoset_name(prefix, deal)][history[i]]
        return reach

    # Deepest information sets first, so later choices are fixed before earlier ones.
    mine = [h for h in DECISION_HISTORIES if player_to_act(h) == br_player]
    for history in sorted(mine, key=len, reverse=True):
        for card in CARDS:
            deals = [d for d in DEALS if d[br_player] == card]
            name = card + ":" + "".join(LETTER[a] for a in history)
            best_a, best_v = None, None
            for a in legal_actions(history):
                v = sum(cf_value(history + (a,), d, opp_reach_to(history, d)) for d in deals)
                if best_v is None or v > best_v:
                    best_a, best_v = a, v
            chosen[name] = best_a

    return sum(cf_value((), deal, Fraction(1)) for deal in DEALS) / len(DEALS)


def best_response_value_bruteforce(probs, br_player):
    """Same quantity, by trying every pure strategy of br_player."""
    mine = sorted(k for k, v in INFOSETS.items() if v[0] == br_player)
    best = None
    for combo in itertools.product(*(INFOSETS[k][1] for k in mine)):
        mixed = dict(probs)
        for k, a in zip(mine, combo):
            mixed[k] = {b: Fraction(1 if b == a else 0) for b in INFOSETS[k][1]}
        v = expected_value_p0(mixed)
        if br_player == 1:
            v = -v
        if best is None or v > best:
            best = v
    return best


def exploitability(probs, br=best_response_value):
    return (br(probs, 0) + br(probs, 1)) / 2


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def to_exact_strings(dist, places=15):
    """Decimal strings that sum to exactly 1."""
    q = Decimal(1).scaleb(-places)
    vals = [Decimal(repr(p)).quantize(q) for p in dist]
    big = max(range(len(vals)), key=lambda i: vals[i])
    vals[big] = Decimal(1) - sum(v for i, v in enumerate(vals) if i != big)
    assert sum(vals) == 1 and all(0 <= v <= 1 for v in vals)
    return [format(v, "f") for v in vals]


def write_strategy(path, avg, iterations):
    probs = {}
    for name in sorted(INFOSETS):
        actions = INFOSETS[name][1]
        probs[name] = dict(zip(actions, to_exact_strings(avg[name])))
    doc = {
        "format": "bluff-strategy-v1",
        "game": "kuhn",
        "producer": "solve_kuhn.py: CFR+ (alternating updates, linear averaging), %d iterations"
        % iterations,
        "probs": probs,
    }
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)
        f.write("\n")


def load_strategy(path):
    with open(path) as f:
        doc = json.load(f)
    probs = {k: {a: Fraction(p) for a, p in v.items()} for k, v in doc["probs"].items()}
    assert set(probs) == set(INFOSETS), "information sets do not match the game"
    for name, (_, actions) in INFOSETS.items():
        assert set(probs[name]) == set(actions), name
        assert sum(probs[name].values()) == 1, name
        assert all(p >= 0 for p in probs[name].values()), name
    return probs


def main():
    iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
    out = sys.argv[2] if len(sys.argv) > 2 else "strategy.json"
    solver = CFRPlus()
    checkpoints = {10 ** k for k in range(1, 9)}
    for t in range(1, iterations + 1):
        solver.iterate(t)
        if t in checkpoints and t < iterations:
            avg = solver.average_strategy()
            probs = {
                k: {a: Fraction(p) for a, p in zip(INFOSETS[k][1], v)} for k, v in avg.items()
            }
            print("iter %8d  exploitability %.3e" % (t, float(exploitability(probs))))
    write_strategy(out, solver.average_strategy(), iterations)

    # Evaluate exactly what was written to disk.
    probs = load_strategy(out)
    value = expected_value_p0(probs)
    br0, br1 = best_response_value(probs, 0), best_response_value(probs, 1)
    expl = (br0 + br1) / 2
    expl_bf = exploitability(probs, best_response_value_bruteforce)
    assert expl == expl_bf, (expl, expl_bf)
    print("wrote %s after %d iterations" % (out, iterations))
    for name in sorted(probs):
        print("  %-5s %s" % (name, {a: round(float(p), 6) for a, p in probs[name].items()}))
    print("value to player 0:       %.12f  (known game value -1/18 = %.12f)" % (float(value), -1 / 18))
    print("best response, player 0: %.12f" % float(br0))
    print("best response, player 1: %.12f" % float(br1))
    print("exploitability:          %.3e  (both methods agree exactly)" % float(expl))
    print(json.dumps({"exploitability": float(expl), "value_p0": float(value)}))


if __name__ == "__main__":
    main()
