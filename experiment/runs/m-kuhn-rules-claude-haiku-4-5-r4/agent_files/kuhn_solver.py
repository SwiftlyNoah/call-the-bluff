import json
import itertools
from typing import Dict, List, Tuple

class KuhnPoker:
    """Kuhn poker solver using counterfactual regret minimization (CFR)."""

    CARDS = ['J', 'Q', 'K']
    CARD_RANK = {'J': 0, 'Q': 1, 'K': 2}

    def __init__(self):
        # infoset_id -> {'regret_sum': {action: float}, 'strategy_sum': {action: float}}
        self.infosets: Dict = {}

    def get_infoset_id(self, card: str, history: List[str]) -> str:
        """Get infoset ID given player's card and action history.

        Only includes 'c' (check) and 'b' (bet) in the name, since fold and call
        lead to terminal states and don't define new infosets.
        """
        filtered = ''.join([a for a in history if a in ['c', 'b']])
        return f"{card}:{filtered}"

    def ensure_infoset(self, infoset_id: str, actions: List[str]):
        """Ensure infoset exists in the dictionary."""
        if infoset_id not in self.infosets:
            self.infosets[infoset_id] = {
                'regret_sum': {action: 0.0 for action in actions},
                'strategy_sum': {action: 0.0 for action in actions},
            }

    def get_legal_actions(self, history: List[str]) -> List[str]:
        """Get legal actions based on action history."""
        if not history:
            return ['check', 'bet']
        last_action = history[-1]
        if last_action == 'c':  # After check, can check or bet
            return ['check', 'bet']
        if last_action == 'b':  # After bet, can fold or call
            return ['fold', 'call']
        return []

    def is_terminal(self, history: List[str]) -> bool:
        """Check if game state is terminal."""
        if not history:
            return False
        # Fold ends the game
        if history[-1] == 'f':
            return True
        # Call ends the game (leads to showdown)
        if history[-1] == 'k':
            return True
        # Two consecutive checks end the game (showdown)
        if len(history) >= 2 and history[-1] == 'c' and history[-2] == 'c':
            return True
        return False

    def get_payoff(self, history: List[str], p0_card: str, p1_card: str) -> float:
        """Get payoff for P0 at terminal state."""
        # Count total chips contributed by each player
        p0_total = 1  # Ante
        p1_total = 1  # Ante

        # Bets and calls add to contributions
        for i, action in enumerate(history):
            player = i % 2  # 0 for P0, 1 for P1
            if action in ['b', 'k']:  # 'b' for bet, 'k' for call
                if player == 0:
                    p0_total += 1
                else:
                    p1_total += 1

        # Determine winner
        if history[-1] == 'f':  # Someone folded
            if len(history) % 2 == 1:  # Odd length means P0 acted last
                # P0 folded, P1 wins
                return -p0_total
            else:
                # P1 folded, P0 wins
                return p1_total

        # Showdown: compare cards
        if self.CARD_RANK[p0_card] > self.CARD_RANK[p1_card]:
            # P0 wins: gains what P1 put in
            return p1_total
        else:
            # P1 wins: P0 loses what they put in
            return -p0_total

    def get_strategy(self, infoset_id: str, actions: List[str]) -> Dict[str, float]:
        """Get current strategy using regret matching."""
        info = self.infosets[infoset_id]
        strategy = {}
        sum_positive_regret = 0.0

        # Regret matching: use positive regrets as weights
        for action in actions:
            regret = max(0, info['regret_sum'][action])
            strategy[action] = regret
            sum_positive_regret += regret

        # Normalize
        if sum_positive_regret > 0:
            for action in actions:
                strategy[action] /= sum_positive_regret
        else:
            # Uniform strategy if no positive regrets
            prob = 1.0 / len(actions)
            for action in actions:
                strategy[action] = prob

        return strategy

    def traverse(self, p0_card: str, p1_card: str, history: List[str],
                 p0_reach: float, p1_reach: float) -> float:
        """
        Traverse game tree using CFR.
        Returns payoff for P0 from this node.
        """
        player = len(history) % 2  # 0 or 1
        my_card = p0_card if player == 0 else p1_card

        # Terminal node
        if self.is_terminal(history):
            return self.get_payoff(history, p0_card, p1_card)

        # Non-terminal node
        actions = self.get_legal_actions(history)
        infoset_id = self.get_infoset_id(my_card, history)
        self.ensure_infoset(infoset_id, actions)

        # Get strategy for this infoset
        strategy = self.get_strategy(infoset_id, actions)

        # Traverse each action and compute values
        node_value = 0.0
        action_values = {}

        action_to_symbol = {
            'check': 'c',
            'bet': 'b',
            'fold': 'f',
            'call': 'k',
        }

        for action in actions:
            symbol = action_to_symbol[action]
            new_history = history + [symbol]

            # Recurse with updated reach probabilities
            if player == 0:
                new_p0_reach = p0_reach * strategy[action]
                action_value = self.traverse(p0_card, p1_card, new_history,
                                            new_p0_reach, p1_reach)
            else:
                new_p1_reach = p1_reach * strategy[action]
                action_value = self.traverse(p0_card, p1_card, new_history,
                                            p0_reach, new_p1_reach)
                # Convert to P1's perspective, then negate back to P0's
                action_value = -action_value

            action_values[action] = action_value
            node_value += strategy[action] * action_value

        # Update regrets and strategy sums
        my_reach = p0_reach if player == 0 else p1_reach
        opp_reach = p1_reach if player == 0 else p0_reach

        for action in actions:
            regret = action_values[action] - node_value
            self.infosets[infoset_id]['regret_sum'][action] += opp_reach * regret
            self.infosets[infoset_id]['strategy_sum'][action] += my_reach * strategy[action]

        # Return value from P0's perspective
        return node_value if player == 0 else -node_value

    def run_cfr(self, num_iterations: int):
        """Run CFR for the specified number of iterations."""
        for iteration in range(num_iterations):
            # Iterate over all possible hands (permutations of 2 cards from 3)
            for p0_card, p1_card in itertools.permutations(self.CARDS, 2):
                # Each hand is equally likely
                self.traverse(p0_card, p1_card, [], 1.0, 1.0)

            if (iteration + 1) % 1000 == 0:
                print(f"Completed iteration {iteration + 1}/{num_iterations}")

    def get_average_strategy(self) -> Dict[str, Dict[str, float]]:
        """Compute average strategy from strategy sums."""
        avg_strategy = {}

        for infoset_id, info in self.infosets.items():
            strategy = {}
            sum_strategy = sum(info['strategy_sum'].values())

            if sum_strategy > 0:
                for action, sum_prob in info['strategy_sum'].items():
                    strategy[action] = sum_prob / sum_strategy
            else:
                # Uniform strategy if no data
                actions = list(info['strategy_sum'].keys())
                prob = 1.0 / len(actions)
                for action in actions:
                    strategy[action] = prob

            avg_strategy[infoset_id] = strategy

        return avg_strategy

    def to_strategy_json(self) -> Dict:
        """Convert to strategy.json format."""
        avg_strategy = self.get_average_strategy()

        # Convert probabilities to strings
        probs = {}
        for infoset_id, strategy in avg_strategy.items():
            probs[infoset_id] = {
                action: str(prob) for action, prob in strategy.items()
            }

        return {
            'format': 'bluff-strategy-v1',
            'game': 'kuhn',
            'producer': 'CFR (Counterfactual Regret Minimization)',
            'probs': probs
        }


if __name__ == '__main__':
    print("Starting Kuhn poker CFR solver...")
    poker = KuhnPoker()

    # Run CFR for 10000 iterations
    poker.run_cfr(10000)
    print("CFR training completed.")

    # Get the strategy and write to file
    strategy = poker.to_strategy_json()
    with open('strategy.json', 'w') as f:
        json.dump(strategy, f, indent=2)

    print("Strategy written to strategy.json")
    print("\nFinal strategy:")
    print(json.dumps(strategy, indent=2))
