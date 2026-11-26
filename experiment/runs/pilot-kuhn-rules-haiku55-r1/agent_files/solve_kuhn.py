"""CFR+ solver for Kuhn poker.

Writes the average strategy of both players to strategy.json (bluff-strategy-v1 format)
and reports the exploitability, computed by exhaustive best response over all pure
strategies (Kuhn has only 6 information sets per player, so this is exact).
"""
import itertools
import json
import sys

CARDS = ["J", "Q", "K"]
RANK = {"J": 0, "Q": 1, "K": 2}
DEALS = [(a, b) for a in CARDS for b in CARDS if a != b]  # 6 equally likely deals
CHANCE = 1.0 / len(DEALS)
LETTER = {"check": "c", "bet": "b"}
TERMINAL_ENDS = ("fold", "call")


def acting_player(hist):
    """Player to act at a non-terminal history (list of action names)."""
    return 0 if len(hist) % 2 == 0 else 1


def is_terminal(hist):
    if not hist:
        return False
    if hist[-1] in TERMINAL_ENDS:
        return True
    return hist == ["check", "check"]


def actions(hist):
    # Facing a bet: fold or call. Otherwise: check or bet.
    if hist and hist[-1] == "bet":
        return ["fold", "call"]
    return ["check", "bet"]


def infoset(card, hist):
    return card + ":" + "".join(LETTER[a] for a in hist)


def payoff_p0(cards, hist):
    """Net chips won by player 0 at a terminal history."""
    last = hist[-1]
    if last == "fold":
        # The player who folded is the one who acted at the previous history.
        folder = acting_player(hist[:-1])
        return -1.0 if folder == 0 else 1.0  # each folder only loses their ante
    c0, c1 = cards
    if RANK[c0] > RANK[c1]:
        winner_p0 = True
    else:
        winner_p0 = False
    # Showdown: 1 chip each if checked through, 2 chips each if a bet was called.
    stake = 2.0 if "bet" in hist else 1.0
    return stake if winner_p0 else -stake


class CFRPlus:
    def __init__(self):
        self.regret = {}   # infoset -> {action: regret}
        self.sumstrat = {}  # infoset -> {action: weighted strategy sum}

    def _get(self, table, key, acts, init=0.0):
        if key not in table:
            table[key] = {a: init for a in acts}
        return table[key]

    def strategy_from_regret(self, key, acts):
        reg = self._get(self.regret, key, acts)
        pos = {a: max(reg[a], 0.0) for a in acts}
        total = sum(pos.values())
        if total > 0:
            return {a: pos[a] / total for a in acts}
        return {a: 1.0 / len(acts) for a in acts}

    def cfr(self, cards, hist, p0, p1, t):
        """Returns the expected payoff to player 0 at this node.

        p0, p1 are the reach probabilities of each player's actions so far
        (chance excluded, which only rescales all regrets by the same constant).
        """
        if is_terminal(hist):
            return payoff_p0(cards, hist)

        pl = acting_player(hist)
        card = cards[pl]
        acts = actions(hist)
        key = infoset(card, hist)
        strat = self.strategy_from_regret(key, acts)

        # Accumulate the average strategy, weighted by the acting player's reach and by t
        # (linear averaging, as in CFR+).
        my_reach = p0 if pl == 0 else p1
        sums = self._get(self.sumstrat, key, acts)
        for a in acts:
            sums[a] += t * my_reach * strat[a]

        util = {}
        node_val = 0.0
        for a in acts:
            if pl == 0:
                u = self.cfr(cards, hist + [a], p0 * strat[a], p1, t)
            else:
                u = self.cfr(cards, hist + [a], p0, p1 * strat[a], t)
            util[a] = u  # payoff to player 0
            node_val += strat[a] * u

        # Regrets are measured in the acting player's own payoff.
        sign = 1.0 if pl == 0 else -1.0
        opp_reach = p1 if pl == 0 else p0
        reg = self._get(self.regret, key, acts)
        for a in acts:
            inst = sign * (util[a] - node_val)
            reg[a] = max(reg[a] + opp_reach * inst, 0.0)  # CFR+: floor regrets at zero
        return node_val

    def train(self, iterations):
        for t in range(1, iterations + 1):
            for cards in DEALS:
                self.cfr(cards, [], 1.0, 1.0, t)

    def average_strategy(self):
        avg = {}
        for key, sums in self.sumstrat.items():
            total = sum(sums.values())
            if total > 0:
                avg[key] = {a: sums[a] / total for a in sums}
            else:
                avg[key] = {a: 1.0 / len(sums) for a in sums}
        return avg


def all_infosets():
    """Enumerate every information set, with its player and legal actions."""
    result = {}
    for cards in DEALS:
        stack = [[]]
        while stack:
            hist = stack.pop()
            if is_terminal(hist):
                continue
            pl = acting_player(hist)
            acts = actions(hist)
            key = infoset(cards[pl], hist)
            result[key] = (pl, acts)
            for a in acts:
                stack.append(hist + [a])
    return result


def expected_value(s0, s1):
    """Expected payoff to player 0 when both players follow behavioral strategies s0, s1.

    s0 and s1 map info set name -> {action: prob}. Each player's strategy only uses
    its own information sets.
    """
    total = 0.0
    for cards in DEALS:
        total += _ev_node(cards, [], 1.0, s0, s1)
    return total * CHANCE


def _ev_node(cards, hist, prob, s0, s1):
    if is_terminal(hist):
        return prob * payoff_p0(cards, hist)
    pl = acting_player(hist)
    key = infoset(cards[pl], hist)
    strat = (s0 if pl == 0 else s1)[key]
    val = 0.0
    for a in actions(hist):
        pa = strat[a]
        if pa == 0.0:
            continue
        val += _ev_node(cards, hist + [a], prob * pa, s0, s1)
    return val


def best_response_value(player, opp_strategy, infosets):
    """Best response value for `player` against opp_strategy, in that player's own payoff.

    Exact: enumerate every pure strategy of the responding player (at most 2^6 = 64 here).
    """
    keys = sorted(k for k, (pl, _) in infosets.items() if pl == player)
    choices = [infosets[k][1] for k in keys]
    best = None
    for combo in itertools.product(*choices):
        pure = {k: {a: (1.0 if a == c else 0.0) for a in infosets[k][1]}
                for k, c in zip(keys, combo)}
        if player == 0:
            s0, s1 = pure, opp_strategy
            v = expected_value(s0, s1)
        else:
            s0, s1 = opp_strategy, pure
            v = -expected_value(s0, s1)
        if best is None or v > best:
            best = v
    return best


def main():
    iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 200000
    solver = CFRPlus()
    solver.train(iterations)
    avg = solver.average_strategy()

    infosets = all_infosets()
    # Split the average strategy by player (strategies are behavioral, one table per player).
    s0 = {k: v for k, v in avg.items() if infosets[k][0] == 0}
    s1 = {k: v for k, v in avg.items() if infosets[k][0] == 1}
    assert set(s0) | set(s1) == set(infosets), "missing information sets"

    br0 = best_response_value(0, s1, infosets)
    br1 = best_response_value(1, s0, infosets)
    exploitability = (br0 + br1) / 2.0
    value_p0 = expected_value(s0, s1)

    probs = {}
    for key in sorted(infosets):
        acts = infosets[key][1]
        probs[key] = {a: repr(float(avg[key][a])) for a in acts}
    out = {
        "format": "bluff-strategy-v1",
        "game": "kuhn",
        "producer": f"CFR+ (linear averaging), {iterations} iterations over all 6 deals, "
                    "averaged strategy for both players",
        "probs": probs,
    }
    with open("strategy.json", "w") as f:
        json.dump(out, f, indent=2)
        f.write("\n")

    print(json.dumps({
        "p0_best_response_value": br0,
        "p1_best_response_value": br1,
        "exploitability": exploitability,
        "value_p0": value_p0,
    }, indent=2))
    for key in sorted(infosets):
        print(key, {a: round(avg[key][a], 4) for a in infosets[key][1]})


if __name__ == "__main__":
    main()
