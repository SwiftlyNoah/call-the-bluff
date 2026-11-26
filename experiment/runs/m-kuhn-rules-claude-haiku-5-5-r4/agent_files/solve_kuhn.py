"""Kuhn poker solver: CFR+ (vanilla chance enumeration, linear averaging) plus an
exact exploitability check by enumerating every pure strategy of each player."""
import itertools
import json

CARDS = "JQK"
RANK = {"J": 0, "Q": 1, "K": 2}
DEALS = [(a, b) for a, b in itertools.permutations(CARDS, 2)]  # 6 equally likely deals

# Non-terminal histories: the letters are the info-set history (c = check/call, b = bet).
ACTIONS = {"": ["check", "bet"], "c": ["check", "bet"], "b": ["fold", "call"], "cb": ["fold", "call"]}
PLAYER = {"": 0, "c": 1, "b": 1, "cb": 0}
NEXT = {
    ("", "check"): "c", ("", "bet"): "b",
    ("c", "check"): "cc", ("c", "bet"): "cb",
    ("b", "fold"): "bf", ("b", "call"): "bc",
    ("cb", "fold"): "cbf", ("cb", "call"): "cbc",
}
LETTER = {"check": "c", "call": "c", "bet": "b", "fold": "f"}
INFOSETS = [c + ":" + h for h in ACTIONS for c in CARDS]  # 12 information sets


def info_name(card, h):
    return f"{card}:{h}"


def terminal_payoff_p0(h, cards):
    """Net chips won by player 0 at terminal history h."""
    showdown = (RANK[cards[0]] > RANK[cards[1]]) - (RANK[cards[0]] < RANK[cards[1]])
    if h == "cc":
        return 1.0 * showdown
    if h == "cbc" or h == "bc":
        return 2.0 * showdown
    if h == "bf":
        return 1.0
    if h == "cbf":
        return -1.0
    raise ValueError(h)


def is_terminal(h):
    return h not in ACTIONS


class CFRPlus:
    def __init__(self):
        self.regret = {I: {a: 0.0 for a in ACTIONS_FOR_INFO[I]} for I in INFOSETS}
        self.strategy_sum = {I: {a: 0.0 for a in ACTIONS_FOR_INFO[I]} for I in INFOSETS}

    def current_strategy(self, I):
        reg = self.regret[I]
        pos = {a: max(r, 0.0) for a, r in reg.items()}
        total = sum(pos.values())
        n = len(pos)
        if total > 0:
            return {a: v / total for a, v in pos.items()}
        return {a: 1.0 / n for a in pos}

    def cfr(self, h, cards, reach, t):
        """Returns expected payoff to player 0 at history h. reach[p] is player p's
        probability of reaching h through its own actions."""
        if is_terminal(h):
            return terminal_payoff_p0(h, cards)
        p = PLAYER[h]
        I = info_name(cards[p], h)
        strat = self.current_strategy(I)
        util = {}
        for a in ACTIONS[h]:
            new_reach = list(reach)
            new_reach[p] *= strat[a]
            util[a] = self.cfr(h + LETTER_NEXT(h, a), cards, new_reach, t)
        node = sum(strat[a] * util[a] for a in ACTIONS[h])
        # Work in the acting player's own utility for regrets.
        sign = 1.0 if p == 0 else -1.0
        opp = reach[1 - p]
        for a in ACTIONS[h]:
            r = sign * (util[a] - node) * opp
            self.regret[I][a] = max(self.regret[I][a] + r, 0.0)  # CFR+ floor
            self.strategy_sum[I][a] += t * reach[p] * strat[a]
        return node

    def train(self, iterations):
        for t in range(1, iterations + 1):
            for cards in DEALS:
                self.cfr("", cards, [1.0, 1.0], t)

    def average_strategy(self):
        avg = {}
        for I in INFOSETS:
            s = self.strategy_sum[I]
            total = sum(s.values())
            if total > 0:
                avg[I] = {a: v / total for a, v in s.items()}
            else:
                avg[I] = {a: 1.0 / len(s) for a in s}
        return avg


def LETTER_NEXT(h, a):
    # Maps an action at history h to its history letter appended to h.
    return LETTER[a]


ACTIONS_FOR_INFO = {}
for _h, _acts in ACTIONS.items():
    for _c in CARDS:
        ACTIONS_FOR_INFO[info_name(_c, _h)] = list(_acts)


def play_value(h, cards, strat):
    """Expected payoff to player 0 from history h under behavioral strategy strat
    (dict: info set -> {action: prob})."""
    if is_terminal(h):
        return terminal_payoff_p0(h, cards)
    p = PLAYER[h]
    I = info_name(cards[p], h)
    total = 0.0
    for a in ACTIONS[h]:
        total += strat[I][a] * play_value(NEXT[(h, a)], cards, strat)
    return total


def pure_strategies(player):
    """All deterministic strategies of one player, as info set -> {action: 0/1}."""
    infos = [I for I in INFOSETS if PLAYER[I.split(":", 1)[1]] == player]
    choices = [ACTIONS_FOR_INFO[I] for I in infos]
    for combo in itertools.product(*choices):
        yield {I: {a: (1.0 if a == pick else 0.0) for a in ACTIONS_FOR_INFO[I]}
               for I, pick in zip(infos, combo)}


def expected_value_p0(strat0, strat1):
    merged = {}
    merged.update(strat0)
    merged.update(strat1)
    return sum(play_value("", cards, merged) for cards in DEALS) / len(DEALS)


def best_response_value_p0(strat1):
    """Best value for player 0 against player 1's behavioral strategy."""
    return max(expected_value_p0(s0, strat1) for s0 in pure_strategies(0))


def best_response_value_p1(strat0):
    """Best value for player 1 (payoff = -player 0's payoff) against player 0's strategy."""
    return max(-expected_value_p0(strat0, s1) for s1 in pure_strategies(1))


def main():
    solver = CFRPlus()
    solver.train(2000000)
    avg = solver.average_strategy()

    # Split into the two players' strategies (each only uses its own info sets).
    strat0 = {I: s for I, s in avg.items() if PLAYER[I.split(":", 1)[1]] == 0}
    strat1 = {I: s for I, s in avg.items() if PLAYER[I.split(":", 1)[1]] == 1}

    value = expected_value_p0(strat0, strat1)
    br0 = best_response_value_p0(strat1)
    br1 = best_response_value_p1(strat0)
    exploitability = (br0 + br1) / 2.0

    probs = {}
    for I in INFOSETS:
        s = avg[I]
        total = sum(s.values())
        probs[I] = {a: repr(v / total) for a, v in s.items()}

    out = {
        "format": "bluff-strategy-v1",
        "game": "kuhn",
        "producer": "CFR+ (2000000 iterations, linear averaging, all chance deals enumerated) in solve_kuhn.py",
        "probs": probs,
    }
    with open("strategy.json", "w") as f:
        json.dump(out, f, indent=2)
        f.write("\n")

    print("Value to P0 (strategy vs strategy):", repr(value))
    print("BR value P0 vs P1:", repr(br0))
    print("BR value P1 vs P0 (raw, P1 payoff):", repr(br1))
    print("Exploitability:", repr(exploitability))
    for I in INFOSETS:
        print(I, {a: round(v, 4) for a, v in avg[I].items()})


if __name__ == "__main__":
    main()
