#!/usr/bin/env python3
import json
from typing import Dict, Any
from collections import defaultdict

def parse_number(s: str) -> float:
    """Parse a number that could be an integer, fraction, or decimal."""
    if '/' in s:
        parts = s.split('/')
        return float(parts[0]) / float(parts[1])
    return float(s)

class CFRSolver:
    def __init__(self, game_data: Dict[str, Any]):
        self.game = game_data
        self.nodes = {node['id']: node for node in game_data['nodes']}
        self.root_id = game_data['root']

        # Build infoset mapping: infoset -> list of action labels
        self.infoset_to_actions = {}
        for node in game_data['nodes']:
            if node['kind'] == 'decision':
                infoset = node['infoset']
                actions = [a['label'] for a in node['actions']]
                if infoset not in self.infoset_to_actions:
                    self.infoset_to_actions[infoset] = actions

        # Regret tracking: infoset -> action -> cumulative regret
        self.cum_regret = defaultdict(lambda: defaultdict(float))

    def get_strategy(self, infoset: str) -> Dict[str, float]:
        """Get current strategy for an infoset using regret matching."""
        actions = self.infoset_to_actions[infoset]
        regrets = self.cum_regret[infoset]

        # Regret matching: take positive regrets, normalize
        positive_regrets = [max(0, regrets[a]) for a in actions]
        total = sum(positive_regrets)

        if total <= 0:
            # No positive regrets, use uniform strategy
            return {a: 1.0 / len(actions) for a in actions}

        return {a: positive_regrets[i] / total for i, a in enumerate(actions)}

    def cfr(self, node_id: int, reach_p0: float, reach_p1: float) -> float:
        """
        Counterfactual regret minimization traversal.
        Returns the expected utility for player 0 at this node.

        reach_p0: probability of reaching this node given player 0's strategy
        reach_p1: probability of reaching this node given player 1's strategy
        """
        node = self.nodes[node_id]

        if node['kind'] == 'terminal':
            return parse_number(node['payoff_p0'])

        if node['kind'] == 'chance':
            # Expected value over chance outcomes
            value = 0.0
            for outcome in node['outcomes']:
                prob = parse_number(outcome['prob'])
                child_id = outcome['child']
                value += prob * self.cfr(child_id, reach_p0, reach_p1)
            return value

        # Decision node
        player = node['player']
        infoset = node['infoset']
        actions = node['actions']
        strategy = self.get_strategy(infoset)

        node_value = 0.0
        action_values = {}

        # Recursively evaluate each action
        for action in actions:
            action_label = action['label']
            child_id = action['child']
            prob = strategy[action_label]

            if player == 0:
                child_value = self.cfr(child_id, reach_p0 * prob, reach_p1)
            else:
                child_value = self.cfr(child_id, reach_p0, reach_p1 * prob)

            action_values[action_label] = child_value
            node_value += prob * child_value

        # Update cumulative regrets
        # Counterfactual reach probability (probability of reaching this infoset from root)
        # times the probability the other player plays to get here
        for action in actions:
            action_label = action['label']
            # Counterfactual value: how much better this action is than the average
            cf_reach = reach_p1 if player == 0 else reach_p0
            regret = (action_values[action_label] - node_value) * cf_reach
            self.cum_regret[infoset][action_label] += regret

        return node_value

    def solve(self, iterations: int = 10000):
        """Run CFR for the specified number of iterations."""
        for i in range(iterations):
            self.cfr(self.root_id, 1.0, 1.0)
            if (i + 1) % 1000 == 0:
                print(f"Completed {i + 1} iterations")

    def get_strategy_from_regrets(self) -> Dict[str, Dict[str, str]]:
        """Extract final strategy from cumulative regrets."""
        result = {}
        for infoset in sorted(self.infoset_to_actions.keys()):
            strategy = self.get_strategy(infoset)
            # Convert to strings for JSON
            result[infoset] = {a: repr(strategy[a]) for a in sorted(strategy.keys())}
        return result

    def compute_exploitability(self) -> float:
        """
        Compute exploitability: best response value for each player against current strategy.
        This is an approximation computed by running best response traversals.
        """
        # Get current strategy
        strategy = {}
        for infoset in self.infoset_to_actions:
            strategy[infoset] = self.get_strategy(infoset)

        game_value = self._compute_value(self.root_id, strategy, 1.0, 1.0)

        # Player 0's best response value against player 1's strategy
        br0_value = self._best_response(self.root_id, 0, strategy, 1.0, 1.0)

        # Player 1's best response value against player 0's strategy (expressed from P0's perspective)
        br1_value = -self._best_response(self.root_id, 1, strategy, 1.0, 1.0)

        # Exploitability = sum of gains from both players deviating
        p0_gain = br0_value - game_value
        p1_gain = br1_value - (-game_value)

        return p0_gain + p1_gain

    def _best_response(self, node_id: int, player: int, strategy: Dict, reach_p0: float, reach_p1: float) -> float:
        """Compute best response value for the given player against the fixed strategy."""
        node = self.nodes[node_id]

        if node['kind'] == 'terminal':
            return parse_number(node['payoff_p0'])

        if node['kind'] == 'chance':
            value = 0.0
            for outcome in node['outcomes']:
                prob = parse_number(outcome['prob'])
                child_id = outcome['child']
                value += prob * self._best_response(child_id, player, strategy, reach_p0, reach_p1)
            return value

        # Decision node
        acting_player = node['player']
        infoset = node['infoset']
        actions = node['actions']

        if acting_player == player:
            # This player chooses the best action
            best_value = float('-inf')
            for action in actions:
                action_label = action['label']
                child_id = action['child']

                if acting_player == 0:
                    child_value = self._best_response(child_id, player, strategy, reach_p0, reach_p1)
                else:
                    child_value = self._best_response(child_id, player, strategy, reach_p0, reach_p1)

                best_value = max(best_value, child_value)
            return best_value
        else:
            # Other player follows their strategy
            strat = strategy[infoset]
            value = 0.0
            for action in actions:
                action_label = action['label']
                child_id = action['child']
                prob = strat[action_label]

                if acting_player == 0:
                    child_value = self._best_response(child_id, player, strategy, reach_p0 * prob, reach_p1)
                else:
                    child_value = self._best_response(child_id, player, strategy, reach_p0, reach_p1 * prob)

                value += prob * child_value
            return value

    def compute_game_value(self) -> float:
        """Compute the value of the game when both players play the converged strategy."""
        strategy = {}
        for infoset in self.infoset_to_actions:
            strategy[infoset] = self.get_strategy(infoset)

        return self._compute_value(self.root_id, strategy, 1.0, 1.0)

    def _compute_value(self, node_id: int, strategy: Dict, reach_p0: float, reach_p1: float) -> float:
        """Compute value when both players play the given strategy."""
        node = self.nodes[node_id]

        if node['kind'] == 'terminal':
            return parse_number(node['payoff_p0'])

        if node['kind'] == 'chance':
            value = 0.0
            for outcome in node['outcomes']:
                prob = parse_number(outcome['prob'])
                child_id = outcome['child']
                value += prob * self._compute_value(child_id, strategy, reach_p0, reach_p1)
            return value

        # Decision node
        player = node['player']
        infoset = node['infoset']
        actions = node['actions']
        strat = strategy[infoset]

        value = 0.0
        for action in actions:
            action_label = action['label']
            child_id = action['child']
            prob = strat[action_label]

            if player == 0:
                child_value = self._compute_value(child_id, strategy, reach_p0 * prob, reach_p1)
            else:
                child_value = self._compute_value(child_id, strategy, reach_p0, reach_p1 * prob)

            value += prob * child_value

        return value

def main():
    # Load game
    with open('game.json') as f:
        game_data = json.load(f)

    # Create and run solver
    solver = CFRSolver(game_data)
    print("Running CFR solver with 100000 iterations...")
    solver.solve(iterations=100000)

    # Get strategy
    strategy = solver.get_strategy_from_regrets()

    # Create output
    output = {
        'format': 'bluff-strategy-v1',
        'game': game_data['name'],
        'producer': 'CFR (Counterfactual Regret Minimization) solver',
        'probs': strategy
    }

    # Write to file
    with open('strategy.json', 'w') as f:
        json.dump(output, f, indent=2)

    print("Strategy written to strategy.json")

    # Compute metrics
    print("\nComputing metrics...")
    game_value = solver.compute_game_value()
    print(f"Game value (player 0 payoff): {game_value}")

    exploitability = solver.compute_exploitability()
    print(f"Exploitability: {exploitability}")

    # Print strategy
    print("\nFinal strategy:")
    for infoset in sorted(strategy.keys()):
        print(f"  {infoset}: {strategy[infoset]}")

if __name__ == '__main__':
    main()
