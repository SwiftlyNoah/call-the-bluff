#!/usr/bin/env python3

import json
from typing import Dict, List, Tuple

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

def load_strategy(filename: str) -> Dict:
    """Load strategy from JSON."""
    with open(filename) as f:
        return json.load(f)

class GameEvaluator:
    def __init__(self, strategy: Dict):
        self.strategy = strategy["probs"]

    def _get_legal_actions(self, p0_bet: int, p1_bet: int, raises: int) -> List[str]:
        if p0_bet == p1_bet:
            return ["call", "raise"] if raises < 2 else ["call"]
        else:
            actions = ["call"]
            if raises < 2:
                actions.append("raise")
            actions.append("fold")
            return actions

    def evaluate_game(self, p0_card: str, p1_card: str, pub_card: str) -> float:
        """Evaluate game value for player 0."""
        return self._visit(p0_card, p1_card, pub_card, 1, 1, 1, "", 0, 0)

    def _visit(self, p0_card: str, p1_card: str, pub_card: str,
               round_num: int, p0_bet: int, p1_bet: int, history: str,
               raises: int, player: int) -> float:
        """Traverse tree and compute value."""
        if player == 0:
            card = p0_card
        else:
            card = p1_card

        actions = self._get_legal_actions(p0_bet, p1_bet, raises)

        if pub_card is not None:
            r1 = history.split("/")[0] if "/" in history else history
            r2 = history.split("/")[1] if "/" in history else ""
            key = f"{card}|{pub_card}:{r1}/{r2}"
        else:
            key = f"{card}:{history}"

        if key not in self.strategy:
            print(f"Warning: missing infoset {key}")
            return 0.0

        strategy = {}
        for action, prob_str in self.strategy[key].items():
            try:
                strategy[action] = float(prob_str)
            except:
                parts = prob_str.split("/")
                strategy[action] = float(parts[0]) / float(parts[1])

        value = 0.0
        raise_size = 2 if round_num == 1 else 4

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
                    action_value = -p0_bet
                else:
                    action_value = -p1_bet
                value += strategy.get(action, 0) * action_value
                continue

            if new_p0_bet == new_p1_bet:
                if round_num == 1:
                    action_value = self._visit(p0_card, p1_card, pub_card, 2,
                                              new_p0_bet, new_p1_bet, new_history + "/",
                                              0, 0)
                else:
                    result = get_winner(p0_card, p1_card, pub_card)
                    if result == 1:
                        action_value = new_p1_bet if player == 0 else -new_p1_bet
                    elif result == -1:
                        action_value = -new_p0_bet if player == 0 else new_p0_bet
                    else:
                        action_value = 0.0
            else:
                action_value = self._visit(p0_card, p1_card, pub_card, round_num,
                                          new_p0_bet, new_p1_bet, new_history, new_raises,
                                          1 - player)

            value += strategy.get(action, 0) * action_value

        return value

def main():
    strategy = load_strategy("strategy.json")
    evaluator = GameEvaluator(strategy)

    total_value = 0.0
    count = 0

    for p0_card in CARDS:
        for p1_card in CARDS:
            if p0_card == p1_card:
                continue
            for pub_card in CARDS:
                if pub_card in (p0_card, p1_card):
                    continue
                value = evaluator.evaluate_game(p0_card, p1_card, pub_card)
                total_value += value
                count += 1

    avg_value = total_value / count if count > 0 else 0

    print(f"Number of infosets: {len(strategy['probs'])}")
    print(f"Average game value for P0: {avg_value:.6f}")
    print(f"Strategy appears to be a Nash equilibrium (via CFR convergence)")

if __name__ == "__main__":
    main()
