"""
Leduc Hold'em solver using CFR+
"""
import json
import random
from collections import defaultdict
from typing import List, Tuple, Dict, Optional
import time


# Card definitions
CARDS = ['Js', 'Jh', 'Qs', 'Qh', 'Ks', 'Kh']
RANKS = {'J': 0, 'Q': 1, 'K': 2}


class LeducState:
    """Represents a state in Leduc Hold'em"""

    def __init__(self, cards: List[str], history: str = '', pot: List[int] = None,
                 round_num: int = 0, public_card: Optional[str] = None):
        self.cards = cards  # [p0_card, p1_card, board_card1, board_card2, ...]
        self.history = history  # Action history: c=call/check, r=raise, f=fold
        self.pot = pot if pot else [1, 1]  # Starting antes
        self.round_num = round_num  # 0 or 1
        self.public_card = public_card

    def get_acting_player(self) -> int:
        """Returns which player acts next (0 or 1)"""
        # Count actions in current round
        if '/' in self.history:
            round_actions = self.history.split('/')[-1]
        else:
            round_actions = self.history.split(':')[-1] if ':' in self.history else self.history

        return len(round_actions) % 2

    def is_terminal(self) -> bool:
        """Check if this is a terminal state"""
        if 'f' in self.history:
            return True

        # Check if betting is complete
        if '/' in self.history:
            # Second round
            round2_actions = self.history.split('/')[-1]
            if len(round2_actions) >= 2 and round2_actions[-1] == 'c':
                return True
            if round2_actions.count('r') >= 2 and len(round2_actions) >= 3 and round2_actions[-1] == 'c':
                return True
        elif ':' in self.history:
            # First round complete, need to check if round ended
            round1_actions = self.history.split(':')[-1]
            if len(round1_actions) >= 2 and round1_actions[-1] == 'c':
                # First round ended, should transition to second round
                return False

        return False

    def get_payoff(self, player: int) -> float:
        """Get payoff for player at terminal state"""
        if not self.is_terminal():
            raise ValueError("Not a terminal state")

        # Check for fold
        if 'f' in self.history:
            folder_action_idx = self.history.index('f')
            if '/' in self.history:
                round2_start = self.history.index('/') + 1
                if folder_action_idx >= round2_start:
                    folder = (folder_action_idx - round2_start) % 2
                    if self.get_acting_player() == 0:
                        folder = 1
                    else:
                        folder = 0
                else:
                    raise ValueError("Fold in first round shouldn't reach here")
            else:
                folder = len(self.history.split(':')[-1]) % 2
                if self.get_acting_player() == 0:
                    folder = 1
                else:
                    folder = 0

            # Winner gets opponent's chips
            winner = 1 - folder
            if player == winner:
                return self.pot[folder]
            else:
                return -self.pot[player]

        # Showdown
        p0_card = self.cards[0]
        p1_card = self.cards[1]
        pub_card = self.public_card

        # Check for pairs
        p0_rank = RANKS[p0_card[0]]
        p1_rank = RANKS[p1_card[0]]
        pub_rank = RANKS[pub_card[0]]

        p0_has_pair = (p0_rank == pub_rank)
        p1_has_pair = (p1_rank == pub_rank)

        if p0_has_pair and not p1_has_pair:
            winner = 0
        elif p1_has_pair and not p0_has_pair:
            winner = 1
        elif p0_has_pair and p1_has_pair:
            # Both have pairs, compare ranks
            if p0_rank > p1_rank:
                winner = 0
            elif p1_rank > p0_rank:
                winner = 1
            else:
                winner = None  # Tie
        else:
            # Neither has pair, compare high cards
            if p0_rank > p1_rank:
                winner = 0
            elif p1_rank > p0_rank:
                winner = 1
            else:
                winner = None  # Tie

        if winner is None:
            return 0.0  # Split pot
        elif winner == player:
            return self.pot[1 - player]
        else:
            return -self.pot[player]

    def get_legal_actions(self) -> List[str]:
        """Get legal actions for current player"""
        if self.is_terminal():
            return []

        # Get current round actions
        if '/' in self.history:
            round_actions = self.history.split('/')[-1]
        else:
            round_actions = self.history.split(':')[-1] if ':' in self.history else self.history

        actions = []

        # Count raises in this round
        num_raises = round_actions.count('r')

        # Check if player owes money
        if not round_actions or round_actions[-1] == 'c':
            # No outstanding bet, can check or raise
            actions.append('call')  # check
            if num_raises < 2:
                actions.append('raise')
        else:
            # Facing a bet/raise
            actions.append('fold')
            actions.append('call')
            if num_raises < 2:
                actions.append('raise')

        return actions

    def apply_action(self, action: str) -> 'LeducState':
        """Apply action and return new state"""
        new_history = self.history + action[0]  # f, c, or r
        new_pot = self.pot.copy()

        player = self.get_acting_player()

        # Get current round actions
        if '/' in self.history:
            round_actions = self.history.split('/')[-1]
            round_num = 1
            raise_size = 4
        else:
            round_actions = self.history.split(':')[-1] if ':' in self.history else self.history
            round_num = 0
            raise_size = 2

        # Update pot based on action
        if action == 'raise':
            # Calculate how much to add
            new_pot[player] += raise_size
            # Also need to call any outstanding bet
            opponent = 1 - player
            if round_actions and round_actions[-1] == 'r':
                # Calling a previous raise
                new_pot[player] += raise_size
        elif action == 'call':
            # Match opponent's bet if any
            if round_actions and round_actions[-1] == 'r':
                opponent = 1 - player
                new_pot[player] = new_pot[opponent]

        # Check if round should end
        if '/' in new_history:
            # Already in second round
            round2_actions = new_history.split('/')[-1]
            if len(round2_actions) >= 2 and round2_actions[-1] == 'c':
                # Second round complete, terminal
                pass
        else:
            # First round
            round1_actions = new_history.split(':')[-1] if ':' in new_history else new_history
            if len(round1_actions) >= 2 and round1_actions[-1] == 'c':
                # First round complete, deal public card and start round 2
                if self.public_card is None:
                    # Deal public card
                    remaining = [c for c in self.cards[2:]]
                    public = remaining[0] if remaining else self.cards[2]
                    new_history = new_history + '/'
                    return LeducState(self.cards, new_history, new_pot, 1, public)

        return LeducState(self.cards, new_history, new_pot, round_num, self.public_card)

    def get_info_set(self, player: int) -> str:
        """Get information set string for player"""
        card = self.cards[player]

        # Get action history from player's perspective
        if '/' in self.history:
            # Second round
            parts = self.history.split('/')
            round1 = parts[0].split(':')[-1] if ':' in parts[0] else parts[0]
            round2 = parts[1]
            return f"{card}|{self.public_card}:{round1}/{round2}"
        else:
            # First round
            actions = self.history.split(':')[-1] if ':' in self.history else self.history
            return f"{card}:{actions}"


class CFRSolver:
    """CFR+ solver for Leduc Hold'em"""

    def __init__(self):
        self.regret_sum = defaultdict(lambda: defaultdict(float))
        self.strategy_sum = defaultdict(lambda: defaultdict(float))
        self.iterations = 0

    def get_strategy(self, info_set: str, actions: List[str]) -> Dict[str, float]:
        """Get current strategy for info set using regret matching"""
        regrets = self.regret_sum[info_set]

        # Regret matching with CFR+ (use max(regret, 0))
        positive_regrets = {a: max(regrets[a], 0) for a in actions}
        total = sum(positive_regrets.values())

        if total > 0:
            strategy = {a: positive_regrets[a] / total for a in actions}
        else:
            # Uniform strategy
            strategy = {a: 1.0 / len(actions) for a in actions}

        return strategy

    def cfr(self, state: LeducState, reach_p0: float, reach_p1: float) -> float:
        """Run CFR recursively"""
        if state.is_terminal():
            return state.get_payoff(0)

        player = state.get_acting_player()
        info_set = state.get_info_set(player)
        actions = state.get_legal_actions()

        # Get current strategy
        strategy = self.get_strategy(info_set, actions)

        # Accumulate strategy (weighted by reach probability)
        reach = reach_p0 if player == 0 else reach_p1
        for action in actions:
            self.strategy_sum[info_set][action] += reach * strategy[action]

        # Compute counterfactual values
        action_values = {}
        node_value = 0.0

        for action in actions:
            next_state = state.apply_action(action)
            if player == 0:
                action_values[action] = -self.cfr(next_state, reach_p0 * strategy[action], reach_p1)
            else:
                action_values[action] = -self.cfr(next_state, reach_p0, reach_p1 * strategy[action])

            node_value += strategy[action] * action_values[action]

        # Update regrets
        for action in actions:
            regret = action_values[action] - node_value
            opponent_reach = reach_p1 if player == 0 else reach_p0
            self.regret_sum[info_set][action] += opponent_reach * regret

        return node_value

    def train(self, iterations: int = 10000):
        """Train the solver"""
        print(f"Training CFR for {iterations} iterations...")
        start_time = time.time()

        for i in range(iterations):
            if (i + 1) % 1000 == 0:
                elapsed = time.time() - start_time
                print(f"Iteration {i + 1}/{iterations} ({elapsed:.1f}s)")

            # Sample a random deal
            deck = CARDS.copy()
            random.shuffle(deck)
            cards = deck  # First 2 are hole cards, rest are for public card

            state = LeducState(cards)
            self.cfr(state, 1.0, 1.0)
            self.iterations += 1

            # CFR+ discount negative regrets
            if (i + 1) % 100 == 0:
                for info_set in self.regret_sum:
                    for action in self.regret_sum[info_set]:
                        self.regret_sum[info_set][action] = max(self.regret_sum[info_set][action], 0)

        elapsed = time.time() - start_time
        print(f"Training complete in {elapsed:.1f}s")

    def get_average_strategy(self) -> Dict[str, Dict[str, str]]:
        """Get average strategy"""
        strategy = {}

        for info_set in self.strategy_sum:
            total = sum(self.strategy_sum[info_set].values())
            if total > 0:
                strategy[info_set] = {
                    action: str(prob / total)
                    for action, prob in self.strategy_sum[info_set].items()
                }
            else:
                # Should not happen, but handle gracefully
                actions = list(self.strategy_sum[info_set].keys())
                strategy[info_set] = {a: str(1.0 / len(actions)) for a in actions}

        return strategy

    def compute_exploitability(self) -> float:
        """Compute exploitability of current average strategy"""
        # This is expensive but useful for checking convergence
        strategy = self.get_average_strategy()

        # Compute best response value for each player
        br_values = []
        for player in [0, 1]:
            value = self.compute_best_response_value(player, strategy)
            br_values.append(value)

        # Exploitability is average of BR values
        return (br_values[0] + br_values[1]) / 2

    def compute_best_response_value(self, player: int, strategy: Dict[str, Dict[str, str]]) -> float:
        """Compute best response value for a player against given strategy"""
        values = []

        # Sample many deals
        for _ in range(1000):
            deck = CARDS.copy()
            random.shuffle(deck)
            state = LeducState(deck)
            value = self.best_response_dfs(state, player, strategy, player)
            values.append(value)

        return sum(values) / len(values)

    def best_response_dfs(self, state: LeducState, br_player: int, strategy: Dict[str, Dict[str, str]],
                          payoff_player: int) -> float:
        """DFS to compute best response value"""
        if state.is_terminal():
            return state.get_payoff(payoff_player)

        player = state.get_acting_player()
        actions = state.get_legal_actions()

        if player == br_player:
            # Maximize for BR player
            best_value = float('-inf')
            for action in actions:
                next_state = state.apply_action(action)
                value = self.best_response_dfs(next_state, br_player, strategy, payoff_player)
                best_value = max(best_value, value)
            return best_value
        else:
            # Opponent plays strategy
            info_set = state.get_info_set(player)
            if info_set not in strategy:
                # Uniform strategy
                action_probs = {a: 1.0 / len(actions) for a in actions}
            else:
                action_probs = {a: float(strategy[info_set][a]) for a in actions}

            value = 0.0
            for action in actions:
                next_state = state.apply_action(action)
                value += action_probs[action] * self.best_response_dfs(next_state, br_player, strategy, payoff_player)
            return value

    def compute_strategy_value(self) -> float:
        """Compute expected value for player 0 when both play average strategy"""
        strategy = self.get_average_strategy()
        values = []

        for _ in range(1000):
            deck = CARDS.copy()
            random.shuffle(deck)
            state = LeducState(deck)
            value = self.evaluate_strategy(state, strategy, 0)
            values.append(value)

        return sum(values) / len(values)

    def evaluate_strategy(self, state: LeducState, strategy: Dict[str, Dict[str, str]],
                          payoff_player: int) -> float:
        """Evaluate strategy from a state"""
        if state.is_terminal():
            return state.get_payoff(payoff_player)

        player = state.get_acting_player()
        info_set = state.get_info_set(player)
        actions = state.get_legal_actions()

        if info_set not in strategy:
            action_probs = {a: 1.0 / len(actions) for a in actions}
        else:
            action_probs = {a: float(strategy[info_set][a]) for a in actions}

        value = 0.0
        for action in actions:
            next_state = state.apply_action(action)
            value += action_probs[action] * self.evaluate_strategy(next_state, strategy, payoff_player)

        return value


def main():
    # Create and train solver
    solver = CFRSolver()
    solver.train(iterations=100000)

    # Get average strategy
    print("\nComputing average strategy...")
    strategy = solver.get_average_strategy()

    # Compute exploitability
    print("Computing exploitability...")
    exploitability = solver.compute_exploitability()
    print(f"Exploitability: {exploitability:.6f}")

    # Compute strategy value
    print("Computing strategy value...")
    value_p0 = solver.compute_strategy_value()
    print(f"Value for P0: {value_p0:.6f}")

    # Save to JSON
    output = {
        "format": "bluff-strategy-v1",
        "game": "leduc",
        "producer": "CFR+ solver, 100k iterations",
        "probs": strategy
    }

    with open('strategy.json', 'w') as f:
        json.dump(output, f, indent=2, sort_keys=True)

    print(f"\nStrategy saved to strategy.json")
    print(f"Total information sets: {len(strategy)}")

    # Print some example strategies
    print("\nExample strategies:")
    for info_set in sorted(strategy.keys())[:5]:
        print(f"  {info_set}: {strategy[info_set]}")

    return exploitability, value_p0


if __name__ == '__main__':
    exploit, value = main()

    # Output final result
    result = {
        "claim_converged": exploit < 0.01,
        "claimed_exploitability": round(exploit, 6),
        "claimed_value_p0": round(value, 6),
        "confidence": "high" if exploit < 0.01 else "medium",
        "notes": f"CFR+ with 100k iterations, exploitability {exploit:.6f}"
    }
    print(json.dumps(result))
