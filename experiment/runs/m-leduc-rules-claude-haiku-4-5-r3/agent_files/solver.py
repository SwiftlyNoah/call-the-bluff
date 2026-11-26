#!/usr/bin/env python3

import json
from typing import Dict, List, Optional, Tuple

CARDS = ["Js", "Jh", "Qs", "Qh", "Ks", "Kh"]

def get_winner(card1: str, card2: str, public_card: str) -> int:
    """Return 1 if card1 wins, -1 if card2 wins, 0 for tie."""
    rank1, rank2 = card1[0], card2[0]
    pub_rank = public_card[0]

    rank_val = {"J": 1, "Q": 2, "K": 3}

    if rank1 == pub_rank and rank2 != pub_rank:
        return 1
    if rank2 == pub_rank and rank1 != pub_rank:
        return -1
    if rank1 == pub_rank and rank2 == pub_rank:
        return 0

    if rank_val[rank1] > rank_val[rank2]:
        return 1
    elif rank_val[rank1] < rank_val[rank2]:
        return -1
    return 0

class CFRSolver:
    def __init__(self):
        self.infosets: Dict[str, Dict] = {}

    def build_infosets(self):
        """Build all infosets by traversing game tree."""
        for p0_card in CARDS:
            for p1_card in CARDS:
                if p0_card == p1_card:
                    continue
                for pub_card in CARDS:
                    if pub_card in (p0_card, p1_card):
                        continue
                    self._traverse_r1(p0_card, p1_card, pub_card, 1, 1, "", 0, 0)

    def _traverse_r1(self, p0_card: str, p1_card: str, pub_card: str,
                     p0_bet: int, p1_bet: int, history: str, raises: int, player: int):
        """Traverse round 1 (before public card revealed)."""
        if player == 0:
            card = p0_card
        else:
            card = p1_card

        actions = self._get_legal_actions(p0_bet, p1_bet, raises)
        key = f"{card}:{history}"

        if key not in self.infosets:
            self.infosets[key] = {
                "actions": list(actions),
                "strategy": {a: 1.0 / len(actions) for a in actions},
                "regrets": {a: 0.0 for a in actions}
            }

        for action in actions:
            raise_size = 2
            new_p0_bet, new_p1_bet = p0_bet, p1_bet
            new_raises = raises

            if action == "raise":
                new_raises += 1
                if player == 0:
                    new_p0_bet += raise_size
                else:
                    new_p1_bet += raise_size
                new_history = history + "r"
            elif action == "call":
                new_p0_bet = max(p0_bet, p1_bet)
                new_p1_bet = max(p0_bet, p1_bet)
                new_history = history + "c"
            else:  # fold
                continue

            if new_p0_bet == new_p1_bet:
                self._traverse_r2(p0_card, p1_card, pub_card, new_p0_bet,
                                 new_p1_bet, new_history + "/", 0, 0)
            else:
                self._traverse_r1(p0_card, p1_card, pub_card, new_p0_bet,
                                 new_p1_bet, new_history, new_raises, 1 - player)

    def _traverse_r2(self, p0_card: str, p1_card: str, pub_card: str,
                     p0_bet: int, p1_bet: int, history: str, raises: int, player: int):
        """Traverse round 2 (after public card revealed)."""
        if player == 0:
            card = p0_card
        else:
            card = p1_card

        actions = self._get_legal_actions(p0_bet, p1_bet, raises)

        r1 = history.split("/")[0]
        r2 = history.split("/")[1] if "/" in history and len(history.split("/")) > 1 else ""
        key = f"{card}|{pub_card}:{r1}/{r2}"

        if key not in self.infosets:
            self.infosets[key] = {
                "actions": list(actions),
                "strategy": {a: 1.0 / len(actions) for a in actions},
                "regrets": {a: 0.0 for a in actions}
            }

        for action in actions:
            raise_size = 4
            new_p0_bet, new_p1_bet = p0_bet, p1_bet
            new_raises = raises

            if action == "raise":
                new_raises += 1
                if player == 0:
                    new_p0_bet += raise_size
                else:
                    new_p1_bet += raise_size
                new_history = history + "r"
            elif action == "call":
                new_p0_bet = max(p0_bet, p1_bet)
                new_p1_bet = max(p0_bet, p1_bet)
                new_history = history + "c"
            else:  # fold
                continue

            if new_p0_bet == new_p1_bet:
                pass
            else:
                self._traverse_r2(p0_card, p1_card, pub_card, new_p0_bet,
                                 new_p1_bet, new_history, new_raises, 1 - player)

    def _get_legal_actions(self, p0_bet: int, p1_bet: int, raises: int) -> List[str]:
        """Get legal actions."""
        if p0_bet == p1_bet:
            return ["call", "raise"] if raises < 2 else ["call"]
        else:
            actions = ["call"]
            if raises < 2:
                actions.append("raise")
            actions.append("fold")
            return actions

    def cfr(self, iterations: int = 1000):
        """Run CFR."""
        self.build_infosets()
        print(f"Built {len(self.infosets)} infosets")

        for t in range(iterations):
            if t % 100 == 0:
                print(f"Iteration {t}/{iterations}")

            for p0_card in CARDS:
                for p1_card in CARDS:
                    if p0_card == p1_card:
                        continue
                    for pub_card in CARDS:
                        if pub_card in (p0_card, p1_card):
                            continue
                        self._cfr_r1(p0_card, p1_card, pub_card, 1, 1, "", 0, 0, 1.0, 1.0)

            self._update_strategies()

    def _cfr_r1(self, p0_card: str, p1_card: str, pub_card: str,
                p0_bet: int, p1_bet: int, history: str, raises: int, player: int,
                p0_reach: float, p1_reach: float) -> float:
        """CFR for round 1."""
        if player == 0:
            card = p0_card
        else:
            card = p1_card

        actions = self._get_legal_actions(p0_bet, p1_bet, raises)
        key = f"{card}:{history}"
        strategy = self.infosets[key]["strategy"]

        values = {}
        node_value = 0.0
        raise_size = 2

        for action in actions:
            new_p0_bet, new_p1_bet = p0_bet, p1_bet
            new_raises = raises

            if action == "raise":
                new_raises += 1
                if player == 0:
                    new_p0_bet += raise_size
                else:
                    new_p1_bet += raise_size
                new_history = history + "r"
            elif action == "call":
                new_p0_bet = max(p0_bet, p1_bet)
                new_p1_bet = max(p0_bet, p1_bet)
                new_history = history + "c"
            else:  # fold
                if player == 0:
                    value = -p0_bet
                else:
                    value = -p1_bet
                values[action] = value
                node_value += strategy[action] * value
                continue

            if new_p0_bet == new_p1_bet:
                new_reach_p0 = p0_reach * strategy[action] if player == 0 else p0_reach
                new_reach_p1 = p1_reach * strategy[action] if player == 1 else p1_reach
                value = self._cfr_r2(p0_card, p1_card, pub_card, new_p0_bet,
                                    new_p1_bet, new_history + "/", 0, 0,
                                    new_reach_p0, new_reach_p1)
            else:
                new_reach_p0 = p0_reach * strategy[action] if player == 0 else p0_reach
                new_reach_p1 = p1_reach * strategy[action] if player == 1 else p1_reach
                value = self._cfr_r1(p0_card, p1_card, pub_card, new_p0_bet,
                                    new_p1_bet, new_history, new_raises, 1 - player,
                                    new_reach_p0, new_reach_p1)

            values[action] = value
            node_value += strategy[action] * value

        opponent_reach = p1_reach if player == 0 else p0_reach

        for action in actions:
            regret = values[action] - node_value
            self.infosets[key]["regrets"][action] += opponent_reach * regret

        return node_value

    def _cfr_r2(self, p0_card: str, p1_card: str, pub_card: str,
                p0_bet: int, p1_bet: int, history: str, raises: int, player: int,
                p0_reach: float, p1_reach: float) -> float:
        """CFR for round 2."""
        if player == 0:
            card = p0_card
        else:
            card = p1_card

        actions = self._get_legal_actions(p0_bet, p1_bet, raises)

        r1 = history.split("/")[0]
        r2 = history.split("/")[1] if "/" in history and len(history.split("/")) > 1 else ""
        key = f"{card}|{pub_card}:{r1}/{r2}"
        strategy = self.infosets[key]["strategy"]

        values = {}
        node_value = 0.0
        raise_size = 4

        for action in actions:
            new_p0_bet, new_p1_bet = p0_bet, p1_bet
            new_raises = raises

            if action == "raise":
                new_raises += 1
                if player == 0:
                    new_p0_bet += raise_size
                else:
                    new_p1_bet += raise_size
                new_history = history + "r"
            elif action == "call":
                new_p0_bet = max(p0_bet, p1_bet)
                new_p1_bet = max(p0_bet, p1_bet)
                new_history = history + "c"
            else:  # fold
                if player == 0:
                    value = -p0_bet
                else:
                    value = -p1_bet
                values[action] = value
                node_value += strategy[action] * value
                continue

            if new_p0_bet == new_p1_bet:
                result = get_winner(p0_card, p1_card, pub_card)
                if result == 1:
                    value = new_p1_bet if player == 0 else -new_p1_bet
                elif result == -1:
                    value = -new_p0_bet if player == 0 else new_p0_bet
                else:
                    value = 0.0
            else:
                new_reach_p0 = p0_reach * strategy[action] if player == 0 else p0_reach
                new_reach_p1 = p1_reach * strategy[action] if player == 1 else p1_reach
                value = self._cfr_r2(p0_card, p1_card, pub_card, new_p0_bet,
                                    new_p1_bet, new_history, new_raises, 1 - player,
                                    new_reach_p0, new_reach_p1)

            values[action] = value
            node_value += strategy[action] * value

        opponent_reach = p1_reach if player == 0 else p0_reach

        for action in actions:
            regret = values[action] - node_value
            self.infosets[key]["regrets"][action] += opponent_reach * regret

        return node_value

    def _update_strategies(self):
        """Update strategies from regrets."""
        for key in self.infosets:
            regrets = self.infosets[key]["regrets"]
            actions = self.infosets[key]["actions"]

            positive_sum = sum(max(r, 0) for r in regrets.values())

            if positive_sum > 0:
                for action in actions:
                    self.infosets[key]["strategy"][action] = max(regrets[action], 0) / positive_sum
            else:
                uniform = 1.0 / len(actions)
                for action in actions:
                    self.infosets[key]["strategy"][action] = uniform

def main():
    solver = CFRSolver()
    solver.cfr(iterations=5000)

    output = {
        "format": "bluff-strategy-v1",
        "game": "leduc",
        "producer": "CFR solver",
        "probs": {}
    }

    for key in sorted(solver.infosets.keys()):
        infoset = solver.infosets[key]
        probs = {}
        for action in infoset["actions"]:
            probs[action] = str(infoset["strategy"][action])
        output["probs"][key] = probs

    with open("strategy.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"Strategy written to strategy.json with {len(output['probs'])} infosets")

if __name__ == "__main__":
    main()
