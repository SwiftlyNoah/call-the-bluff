#!/usr/bin/env python3
"""Analyze convergence of CFR solver."""

import json
from fractions import Fraction
from collections import defaultdict

def parse_number(s):
    """Parse a number from string to Fraction."""
    if '/' in s:
        parts = s.split('/')
        return Fraction(int(parts[0]), int(parts[1]))
    else:
        return Fraction(s)

class CFRSolver:
    def __init__(self, game_path):
        with open(game_path) as f:
            self.game = json.load(f)

        self.nodes = self.game['nodes']
        self.root = self.game['root']
        self.name = self.game['name']

        # Build node lookup
        self.node_map = {node['id']: node for node in self.nodes}

        # Regrets and strategy sums
        self.regrets = defaultdict(lambda: defaultdict(float))
        self.strategy_sum = defaultdict(lambda: defaultdict(float))

        # Collect all infosets
        self.infosets = {}
        for node in self.nodes:
            if node['kind'] == 'decision':
                infoset = node['infoset']
                if infoset not in self.infosets:
                    self.infosets[infoset] = {
                        'player': node['player'],
                        'actions': [a['label'] for a in node['actions']]
                    }

        # For convergence tracking
        self.game_values = []

    def get_strategy(self, infoset):
        """Get current strategy for an infoset using regret matching."""
        actions = self.infosets[infoset]['actions']
        regrets = self.regrets[infoset]

        # Regret matching: use positive regrets
        strategy = {}
        total_positive_regret = 0.0

        for action in actions:
            regret = max(0.0, regrets.get(action, 0.0))
            strategy[action] = regret
            total_positive_regret += regret

        # Normalize
        if total_positive_regret > 0:
            for action in actions:
                strategy[action] /= total_positive_regret
        else:
            # Uniform strategy
            uniform_prob = 1.0 / len(actions)
            for action in actions:
                strategy[action] = uniform_prob

        return strategy

    def cfr(self, node_id, reach_probs, iteration):
        """Counterfactual Regret Minimization."""
        node = self.node_map[node_id]

        if node['kind'] == 'terminal':
            payoff = parse_number(node['payoff_p0'])
            return float(payoff)

        if node['kind'] == 'chance':
            # Chance node: take expectation over outcomes
            value = 0.0
            for outcome in node['outcomes']:
                prob = parse_number(outcome['prob'])
                child_id = outcome['child']
                child_value = self.cfr(child_id, reach_probs.copy(), iteration)
                value += float(prob) * child_value
            return value

        # Decision node
        player = node['player']
        infoset = node['infoset']
        actions = node['actions']

        # Get current strategy
        strategy = self.get_strategy(infoset)

        # Compute value for each action
        action_values = {}
        node_value = 0.0

        for action in actions:
            action_obj = next(a for a in actions if a['label'] == action['label'])
            child_id = action_obj['child']

            # Update reach probability
            new_reach_probs = reach_probs.copy()
            new_reach_probs[player] *= strategy[action['label']]

            # Recursively compute value
            child_value = self.cfr(child_id, new_reach_probs, iteration)
            action_values[action['label']] = child_value
            node_value += strategy[action['label']] * child_value

        # Update regrets
        opponent_reach = reach_probs[1 - player]

        for action in actions:
            action_label = action['label']
            regret = action_values[action_label] - node_value
            # Weight by opponent's reach probability
            self.regrets[infoset][action_label] += opponent_reach * regret
            # Accumulate strategy
            self.strategy_sum[infoset][action_label] += reach_probs[player] * strategy[action_label]

        return node_value

    def solve(self, iterations=2000):
        """Run CFR+ for the specified number of iterations."""
        print(f"Running CFR+ for {iterations} iterations...")
        for i in range(iterations):
            if (i + 1) % 500 == 0:
                print(f"  Iteration {i + 1}/{iterations}")
            reach_probs = {0: 1.0, 1: 1.0}
            game_value = self.cfr(self.root, reach_probs, i)
            self.game_values.append(game_value)
        print("Done!")

def main():
    solver = CFRSolver('game.json')
    solver.solve(iterations=2000)

    print("\n" + "=" * 60)
    print("CONVERGENCE ANALYSIS")
    print("=" * 60)

    # Check game value convergence
    print("\nGame value convergence (sample every 200 iterations):")
    for i in range(0, len(solver.game_values), 200):
        print(f"  Iteration {i+1:4d}: {solver.game_values[i]:10.6f}")

    # Check for stability in last iterations
    last_values = solver.game_values[-100:]
    min_val = min(last_values)
    max_val = max(last_values)
    mean_val = sum(last_values) / len(last_values)
    std_val = (sum((v - mean_val)**2 for v in last_values) / len(last_values)) ** 0.5

    print(f"\nLast 100 iterations:")
    print(f"  Min:    {min_val:.6f}")
    print(f"  Max:    {max_val:.6f}")
    print(f"  Mean:   {mean_val:.6f}")
    print(f"  StdDev: {std_val:.9f}")
    print(f"  Range:  {max_val - min_val:.9f}")

    # Analyze regrets magnitude
    print("\nRegret magnitude analysis:")
    all_regrets = []
    for infoset, regrets in solver.regrets.items():
        for action, regret in regrets.items():
            all_regrets.append(abs(regret))

    if all_regrets:
        all_regrets.sort()
        print(f"  Number of regrets: {len(all_regrets)}")
        print(f"  Max regret magnitude: {all_regrets[-1]:.6f}")
        print(f"  Median regret: {all_regrets[len(all_regrets)//2]:.6f}")
        print(f"  Mean regret magnitude: {sum(all_regrets)/len(all_regrets):.6f}")

    # Exploitability estimate
    print("\nExploitability estimate:")
    positive_regrets = []
    for infoset, regrets in solver.regrets.items():
        for action, regret in regrets.items():
            if regret > 0:
                positive_regrets.append(regret)

    if positive_regrets:
        max_positive = max(positive_regrets)
        avg_positive = sum(positive_regrets) / len(positive_regrets)
        print(f"  Positive regrets: {len(positive_regrets)} out of {len(all_regrets)}")
        print(f"  Max positive regret: {max_positive:.6f}")
        print(f"  Avg positive regret: {avg_positive:.6f}")
        # Very rough proxy for exploitability (not rigorous)
        rough_exploit = max_positive / 20.0
        print(f"  (Rough exploitability estimate: {rough_exploit:.6f})")
    else:
        print(f"  No positive regrets found")

    print("\n" + "=" * 60)
    print(f"SUMMARY")
    print("=" * 60)
    print(f"Final game value (Player 0): {mean_val:.6f}")
    print(f"Convergence: {'STRONG' if std_val < 0.0001 else 'MODERATE' if std_val < 0.001 else 'WEAK'}")
    print("=" * 60)

if __name__ == '__main__':
    main()
