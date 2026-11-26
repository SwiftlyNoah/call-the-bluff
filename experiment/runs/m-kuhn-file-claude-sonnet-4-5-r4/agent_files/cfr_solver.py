#!/usr/bin/env python3
"""
CFR solver with separate traversals per player.
"""

import json
from collections import defaultdict
from typing import Dict


def parse_rational(s: str) -> float:
    """Parse a rational number string to float."""
    if '/' in s:
        num, denom = s.split('/')
        return float(num) / float(denom)
    return float(s)


class CFRSolver:
    """Implements CFR with player-specific traversals."""

    def __init__(self, game_data: dict):
        self.game_name = game_data['name']
        self.nodes = {node['id']: node for node in game_data['nodes']}
        self.root_id = game_data['root']

        # CFR data structures
        self.regret_sum = defaultdict(lambda: defaultdict(float))
        self.strategy_sum = defaultdict(lambda: defaultdict(float))

        # Build infoset mapping
        self.infosets = {}
        for node in self.nodes.values():
            if node['kind'] == 'decision':
                infoset = node['infoset']
                if infoset not in self.infosets:
                    actions = [a['label'] for a in node['actions']]
                    self.infosets[infoset] = (node['player'], actions)

    def get_strategy(self, infoset: str) -> Dict[str, float]:
        """Compute current strategy using regret matching."""
        _, actions = self.infosets[infoset]

        # Positive regrets only
        pos_regrets = {a: max(0, self.regret_sum[infoset][a]) for a in actions}
        total = sum(pos_regrets.values())

        if total > 0:
            return {a: pos_regrets[a] / total for a in actions}
        else:
            return {a: 1.0 / len(actions) for a in actions}

    def cfr(self, node_id: int, traversing_player: int, p0: float, p1: float) -> float:
        """
        Run CFR from this node for a specific traversing player.
        Only update regrets for the traversing player.
        Returns: utility for player 0.
        """
        node = self.nodes[node_id]
        kind = node['kind']

        if kind == 'terminal':
            return parse_rational(node['payoff_p0'])

        if kind == 'chance':
            util = 0.0
            for outcome in node['outcomes']:
                prob = parse_rational(outcome['prob'])
                util += prob * self.cfr(outcome['child'], traversing_player, p0, p1)
            return util

        # Decision node
        player = node['player']
        infoset = node['infoset']
        actions = [a['label'] for a in node['actions']]

        strategy = self.get_strategy(infoset)

        # Compute utility for each action
        utils = {}
        for action_dict in node['actions']:
            action = action_dict['label']
            prob = strategy[action]

            if player == 0:
                utils[action] = self.cfr(action_dict['child'], traversing_player, p0 * prob, p1)
            else:
                utils[action] = self.cfr(action_dict['child'], traversing_player, p0, p1 * prob)

        # Node utility
        node_util = sum(strategy[a] * utils[a] for a in actions)

        # Only update regrets for the traversing player
        if player == traversing_player:
            # Counterfactual reach: opponent's reach
            cf_reach = p1 if player == 0 else p0

            # For player 1, negate utilities (they're from P0's perspective)
            if player == 1:
                for action in actions:
                    regret = (-utils[action]) - (-node_util)
                    self.regret_sum[infoset][action] += cf_reach * regret
                    # CFR+: floor regrets at 0
                    self.regret_sum[infoset][action] = max(0.0, self.regret_sum[infoset][action])
            else:
                for action in actions:
                    regret = utils[action] - node_util
                    self.regret_sum[infoset][action] += cf_reach * regret
                    # CFR+: floor regrets at 0
                    self.regret_sum[infoset][action] = max(0.0, self.regret_sum[infoset][action])

            # Update strategy sum (weighted by own reach)
            own_reach = p0 if player == 0 else p1
            for action in actions:
                self.strategy_sum[infoset][action] += own_reach * strategy[action]

        return node_util

    def train(self, iterations: int):
        """Run CFR for given iterations."""
        for i in range(iterations):
            # Traverse once for each player
            for player in [0, 1]:
                self.cfr(self.root_id, player, 1.0, 1.0)

            if (i + 1) % 100000 == 0:
                exploit = self.compute_exploitability()
                print(f"Iteration {i + 1}/{iterations}, Exploitability: {exploit:.8f}")

    def get_average_strategy(self) -> Dict[str, Dict[str, str]]:
        """Return average strategy."""
        result = {}
        for infoset, (player, actions) in self.infosets.items():
            total = sum(self.strategy_sum[infoset][a] for a in actions)
            if total > 0:
                result[infoset] = {
                    a: repr(self.strategy_sum[infoset][a] / total)
                    for a in actions
                }
            else:
                n = len(actions)
                result[infoset] = {a: repr(1.0/n) for a in actions}
        return result

    def eval_strategy(self, strategy: dict, node_id: int) -> float:
        """Evaluate a strategy at a node."""
        node = self.nodes[node_id]
        kind = node['kind']

        if kind == 'terminal':
            return parse_rational(node['payoff_p0'])

        if kind == 'chance':
            util = 0.0
            for outcome in node['outcomes']:
                prob = parse_rational(outcome['prob'])
                util += prob * self.eval_strategy(strategy, outcome['child'])
            return util

        # Decision
        infoset = node['infoset']
        probs = strategy[infoset]

        util = 0.0
        for action_dict in node['actions']:
            action = action_dict['label']
            util += probs[action] * self.eval_strategy(strategy, action_dict['child'])
        return util

    def compute_exploitability(self) -> float:
        """Compute exploitability of average strategy."""
        avg_strat = self.get_average_strategy()
        strat = {infoset: {a: float(avg_strat[infoset][a]) for a in avg_strat[infoset]}
                 for infoset in avg_strat}

        # Best response for P0
        br0_val = self.best_response(0, strat, self.root_id)
        # Best response for P1
        br1_val = self.best_response(1, strat, self.root_id)

        # Strategy value
        strat_val = self.eval_strategy(strat, self.root_id)

        # Exploitability
        return ((br0_val - strat_val) + (br1_val - (-strat_val))) / 2

    def best_response(self, player: int, opp_strat: dict, node_id: int) -> float:
        """Compute best response value for player."""
        node = self.nodes[node_id]
        kind = node['kind']

        if kind == 'terminal':
            payoff = parse_rational(node['payoff_p0'])
            return payoff if player == 0 else -payoff

        if kind == 'chance':
            val = 0.0
            for outcome in node['outcomes']:
                prob = parse_rational(outcome['prob'])
                val += prob * self.best_response(player, opp_strat, outcome['child'])
            return val

        # Decision
        if node['player'] == player:
            # Best response: maximize
            return max(
                self.best_response(player, opp_strat, a['child'])
                for a in node['actions']
            )
        else:
            # Opponent plays their strategy
            infoset = node['infoset']
            probs = opp_strat[infoset]
            return sum(
                probs[a['label']] * self.best_response(player, opp_strat, a['child'])
                for a in node['actions']
            )


def main():
    with open('game.json', 'r') as f:
        game_data = json.load(f)

    print(f"Loaded game: {game_data['name']}")

    solver = CFRSolver(game_data)
    print(f"Found {len(solver.infosets)} information sets\n")

    print("Training CFR+ (1M iterations)...")
    solver.train(iterations=1000000)

    avg_strategy = solver.get_average_strategy()

    # Convert for evaluation
    strat = {infoset: {a: float(avg_strategy[infoset][a]) for a in avg_strategy[infoset]}
             for infoset in avg_strategy}

    expected_value = solver.eval_strategy(strat, solver.root_id)
    exploitability = solver.compute_exploitability()

    print(f"\nExpected value for P0: {expected_value:.8f}")
    print(f"Exploitability: {exploitability:.8f}")

    output = {
        "format": "bluff-strategy-v1",
        "game": game_data['name'],
        "producer": "CFR+ solver (1M iterations)",
        "probs": avg_strategy
    }

    with open('strategy.json', 'w') as f:
        json.dump(output, f, indent=1)

    print(f"\nStrategy written to strategy.json")
    print("\nKey strategies:")
    for infoset in sorted([k for k in avg_strategy.keys() if ':' not in k[2:]]):
        print(f"  {infoset}: {avg_strategy[infoset]}")


if __name__ == '__main__':
    main()
