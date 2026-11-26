"""
Leduc Hold'em solver using CFR+ (v2 with improved game logic)
"""
import json
import random
from collections import defaultdict
from typing import List, Tuple, Dict, Optional
import time


# Card definitions
CARDS = ['Js', 'Jh', 'Qs', 'Qh', 'Ks', 'Kh']
RANKS = {'J': 0, 'Q': 1, 'K': 2}


class LeducGame:
    """Represents a game of Leduc Hold'em"""

    def __init__(self, cards: List[str]):
        """Initialize with shuffled deck (first 2 are hole cards, next is board)"""
        self.cards = cards
        self.reset()

    def reset(self):
        """Reset to initial state"""
        self.history = ""
        self.round = 0  # 0 = first betting round, 1 = second betting round
        self.pot = [1, 1]  # Starting antes
        self.pending = [0, 0]  # Pending bets in current round
        self.board = None

    def player_to_act(self) -> int:
        """Get player whose turn it is"""
        # Count actions in current betting round
        if '/' in self.history:
            actions = self.history.split('/')[-1]
        else:
            actions = self.history
        return len(actions) % 2

    def is_terminal(self) -> bool:
        """Check if game is over"""
        if 'f' in self.history:
            return True

        # Check if betting round is complete
        if '/' in self.history:
            # In second round
            actions = self.history.split('/')[-1]
        else:
            # In first round
            actions = self.history

        if len(actions) < 2:
            return False

        # Check if betting is settled (last action was call/check after at least one action)
        if actions[-1] != 'c':
            return False

        # If we're in second round and betting is settled, game is over
        if '/' in self.history:
            return True

        # First round is over, but game continues to second round
        return False

    def get_legal_actions(self) -> List[str]:
        """Get legal actions for current player"""
        if self.is_terminal():
            return []

        # Get current round actions
        if '/' in self.history:
            actions = self.history.split('/')[-1]
        else:
            actions = self.history

        player = self.player_to_act()
        opponent = 1 - player

        legal = []

        # Count raises this round
        num_raises = actions.count('r')

        # Determine if player faces a bet
        faces_bet = (len(actions) > 0 and actions[-1] == 'r')

        if faces_bet:
            legal.append('fold')
            legal.append('call')
            if num_raises < 2:
                legal.append('raise')
        else:
            legal.append('call')  # check
            if num_raises < 2:
                legal.append('raise')

        return legal

    def apply_action(self, action: str):
        """Apply action and update state"""
        player = self.player_to_act()
        opponent = 1 - player

        # Determine round and raise size
        if '/' in self.history:
            raise_size = 4
        else:
            raise_size = 2

        # Update history
        self.history += action[0]

        # Update pending bets
        if action == 'raise':
            # Match opponent's bet and add raise
            amount_to_call = self.pending[opponent] - self.pending[player]
            self.pending[player] += amount_to_call + raise_size
        elif action == 'call':
            # Match opponent's bet
            amount_to_call = self.pending[opponent] - self.pending[player]
            self.pending[player] += amount_to_call

        # Check if betting round is complete
        if '/' in self.history:
            actions = self.history.split('/')[-1]
        else:
            actions = self.history

        # Round ends when last action is call and at least 2 actions
        if len(actions) >= 2 and actions[-1] == 'c' and not self.is_terminal():
            # Settle bets
            self.pot[0] += self.pending[0]
            self.pot[1] += self.pending[1]
            self.pending = [0, 0]

            # If first round just ended, deal board and start second round
            if '/' not in self.history:
                self.board = self.cards[2]
                self.history += '/'
                self.round = 1

    def get_payoff(self, player: int) -> float:
        """Get payoff for player (must be terminal)"""
        if not self.is_terminal():
            raise ValueError("Not terminal")

        # Handle fold
        if 'f' in self.history:
            if '/' in self.history:
                actions = self.history.split('/')[-1]
            else:
                actions = self.history

            # Player who folded is determined by action count
            folder = (len(actions) - 1) % 2
            winner = 1 - folder

            # Settle remaining pending bets
            pot = self.pot[:]
            pot[0] += self.pending[0]
            pot[1] += self.pending[1]

            # Net payoff
            if player == winner:
                return pot[folder]
            else:
                return -pot[player]

        # Showdown
        # Settle all bets first
        pot = self.pot[:]
        pot[0] += self.pending[0]
        pot[1] += self.pending[1]

        # Determine winner
        p0_card = self.cards[0]
        p1_card = self.cards[1]
        board = self.board

        p0_rank = RANKS[p0_card[0]]
        p1_rank = RANKS[p1_card[0]]
        board_rank = RANKS[board[0]]

        p0_pair = (p0_rank == board_rank)
        p1_pair = (p1_rank == board_rank)

        if p0_pair and not p1_pair:
            winner = 0
        elif p1_pair and not p0_pair:
            winner = 1
        elif p0_pair and p1_pair:
            # Both have pairs, higher rank wins
            if p0_rank > p1_rank:
                winner = 0
            elif p1_rank > p0_rank:
                winner = 1
            else:
                winner = None  # Tie
        else:
            # No pairs, higher card wins
            if p0_rank > p1_rank:
                winner = 0
            elif p1_rank > p0_rank:
                winner = 1
            else:
                winner = None  # Tie

        if winner is None:
            return 0.0
        elif winner == player:
            return pot[1 - player]
        else:
            return -pot[player]

    def get_info_set(self, player: int) -> str:
        """Get information set for player"""
        card = self.cards[player]

        if '/' in self.history:
            # Second round
            parts = self.history.split('/')
            round1 = parts[0]
            round2 = parts[1]
            return f"{card}|{self.board}:{round1}/{round2}"
        else:
            # First round
            return f"{card}:{self.history}"


class CFRSolver:
    """CFR+ solver"""

    def __init__(self):
        self.regret_sum = defaultdict(lambda: defaultdict(float))
        self.strategy_sum = defaultdict(lambda: defaultdict(float))
        self.iterations = 0

    def get_strategy(self, info_set: str, actions: List[str]) -> Dict[str, float]:
        """Get current strategy using regret matching"""
        regrets = self.regret_sum[info_set]

        # CFR+: use max(regret, 0)
        positive_regrets = {a: max(regrets.get(a, 0), 0) for a in actions}
        total = sum(positive_regrets.values())

        if total > 0:
            return {a: positive_regrets[a] / total for a in actions}
        else:
            return {a: 1.0 / len(actions) for a in actions}

    def cfr(self, game: LeducGame, reach_p0: float, reach_p1: float) -> float:
        """CFR recursion"""
        if game.is_terminal():
            return game.get_payoff(0)

        player = game.player_to_act()
        info_set = game.get_info_set(player)
        actions = game.get_legal_actions()

        strategy = self.get_strategy(info_set, actions)

        # Add to strategy sum
        reach = reach_p0 if player == 0 else reach_p1
        for a in actions:
            self.strategy_sum[info_set][a] += reach * strategy[a]

        # Compute action values
        action_values = {}
        node_value = 0.0

        for action in actions:
            # Save state
            old_history = game.history
            old_pot = game.pot[:]
            old_pending = game.pending[:]
            old_board = game.board
            old_round = game.round

            # Apply action
            game.apply_action(action)

            # Recurse
            if player == 0:
                action_values[action] = -self.cfr(game, reach_p0 * strategy[action], reach_p1)
            else:
                action_values[action] = -self.cfr(game, reach_p0, reach_p1 * strategy[action])

            node_value += strategy[action] * action_values[action]

            # Restore state
            game.history = old_history
            game.pot = old_pot
            game.pending = old_pending
            game.board = old_board
            game.round = old_round

        # Update regrets
        opp_reach = reach_p1 if player == 0 else reach_p0
        for action in actions:
            regret = action_values[action] - node_value
            self.regret_sum[info_set][action] += opp_reach * regret

        return node_value

    def train(self, iterations: int):
        """Train the solver"""
        print(f"Training for {iterations} iterations...")
        start = time.time()

        for i in range(iterations):
            if (i + 1) % 1000 == 0:
                print(f"Iteration {i + 1}/{iterations} ({time.time() - start:.1f}s)")

            # Random deal
            deck = CARDS[:]
            random.shuffle(deck)

            game = LeducGame(deck)
            self.cfr(game, 1.0, 1.0)

            # CFR+ discounting
            if (i + 1) % 100 == 0:
                for info_set in self.regret_sum:
                    for action in self.regret_sum[info_set]:
                        self.regret_sum[info_set][action] = max(0, self.regret_sum[info_set][action])

        print(f"Training complete in {time.time() - start:.1f}s")

    def get_average_strategy(self) -> Dict[str, Dict[str, str]]:
        """Extract average strategy"""
        strategy = {}

        for info_set in self.strategy_sum:
            total = sum(self.strategy_sum[info_set].values())
            if total > 0:
                strategy[info_set] = {
                    a: str(p / total)
                    for a, p in self.strategy_sum[info_set].items()
                }
            else:
                actions = list(self.strategy_sum[info_set].keys())
                strategy[info_set] = {a: str(1.0 / len(actions)) for a in actions}

        return strategy

    def compute_exploitability(self) -> float:
        """Compute exploitability"""
        strategy = self.get_average_strategy()

        values = []
        for player in [0, 1]:
            value = self.best_response_value(player, strategy)
            values.append(value)

        return (values[0] + values[1]) / 2

    def best_response_value(self, br_player: int, strategy: Dict) -> float:
        """Compute best response value"""
        total = 0.0
        count = 0

        for _ in range(1000):
            deck = CARDS[:]
            random.shuffle(deck)
            game = LeducGame(deck)
            value = self.br_dfs(game, br_player, strategy, br_player)
            total += value
            count += 1

        return total / count

    def br_dfs(self, game: LeducGame, br_player: int, strategy: Dict,
               payoff_player: int) -> float:
        """Best response DFS"""
        if game.is_terminal():
            return game.get_payoff(payoff_player)

        player = game.player_to_act()
        actions = game.get_legal_actions()

        if player == br_player:
            # Maximize
            best = float('-inf')
            for action in actions:
                old_history = game.history
                old_pot = game.pot[:]
                old_pending = game.pending[:]
                old_board = game.board
                old_round = game.round

                game.apply_action(action)
                value = self.br_dfs(game, br_player, strategy, payoff_player)
                best = max(best, value)

                game.history = old_history
                game.pot = old_pot
                game.pending = old_pending
                game.board = old_board
                game.round = old_round

            return best
        else:
            # Opponent plays strategy
            info_set = game.get_info_set(player)
            if info_set not in strategy:
                probs = {a: 1.0 / len(actions) for a in actions}
            else:
                probs = {a: float(strategy[info_set][a]) for a in actions}

            value = 0.0
            for action in actions:
                old_history = game.history
                old_pot = game.pot[:]
                old_pending = game.pending[:]
                old_board = game.board
                old_round = game.round

                game.apply_action(action)
                value += probs[action] * self.br_dfs(game, br_player, strategy, payoff_player)

                game.history = old_history
                game.pot = old_pot
                game.pending = old_pending
                game.board = old_board
                game.round = old_round

            return value

    def compute_game_value(self) -> float:
        """Compute game value for P0"""
        strategy = self.get_average_strategy()

        total = 0.0
        for _ in range(1000):
            deck = CARDS[:]
            random.shuffle(deck)
            game = LeducGame(deck)
            value = self.eval_strategy(game, strategy, 0)
            total += value

        return total / 1000

    def eval_strategy(self, game: LeducGame, strategy: Dict, payoff_player: int) -> float:
        """Evaluate strategy"""
        if game.is_terminal():
            return game.get_payoff(payoff_player)

        player = game.player_to_act()
        info_set = game.get_info_set(player)
        actions = game.get_legal_actions()

        if info_set not in strategy:
            probs = {a: 1.0 / len(actions) for a in actions}
        else:
            probs = {a: float(strategy[info_set][a]) for a in actions}

        value = 0.0
        for action in actions:
            old_history = game.history
            old_pot = game.pot[:]
            old_pending = game.pending[:]
            old_board = game.board
            old_round = game.round

            game.apply_action(action)
            value += probs[action] * self.eval_strategy(game, strategy, payoff_player)

            game.history = old_history
            game.pot = old_pot
            game.pending = old_pending
            game.board = old_board
            game.round = old_round

        return value


def main():
    solver = CFRSolver()
    solver.train(100000)

    print("\nComputing metrics...")
    strategy = solver.get_average_strategy()

    print("Computing exploitability...")
    exploit = solver.compute_exploitability()
    print(f"Exploitability: {exploit:.6f}")

    print("Computing game value...")
    value = solver.compute_game_value()
    print(f"P0 value: {value:.6f}")

    # Save strategy
    output = {
        "format": "bluff-strategy-v1",
        "game": "leduc",
        "producer": "CFR+ solver, 100k iterations",
        "probs": strategy
    }

    with open('strategy.json', 'w') as f:
        json.dump(output, f, indent=2, sort_keys=True)

    print(f"\nSaved to strategy.json ({len(strategy)} info sets)")

    return exploit, value


if __name__ == '__main__':
    exploit, value = main()

    result = {
        "claim_converged": exploit < 0.01,
        "claimed_exploitability": round(exploit, 6),
        "claimed_value_p0": round(value, 6),
        "confidence": "high" if exploit < 0.01 else "medium",
        "notes": f"CFR+ 100k iterations, exploitability {exploit:.6f}"
    }
    print(json.dumps(result))
