#!/usr/bin/env python3
import json
from collections import defaultdict

class CFRSolver:
    def __init__(self, game_data):
        self.game_data = game_data
        self.nodes = {node['id']: node for node in game_data['nodes']}
        self.root = game_data['root']
        self.name = game_data['name']

        # Strategy data structures
        self.cumulative_regrets = defaultdict(lambda: defaultdict(float))
        self.action_counts = defaultdict(lambda: defaultdict(int))

    def parse_number(self, s):
        """Parse a number string (int, float, or fraction) to float."""
        if '/' in str(s):
            parts = str(s).split('/')
            return float(parts[0]) / float(parts[1])
        return float(s)

    def get_strategy(self, infoset_name, num_actions):
        """Get current strategy for an information set using regret matching."""
        strategy = {}
        regrets = self.cumulative_regrets[infoset_name]

        # Collect positive regrets
        sum_positive = 0
        for i in range(num_actions):
            r = max(0, regrets.get(i, 0))
            sum_positive += r

        # Regret matching: use positive regrets as strategy
        if sum_positive > 0:
            for i in range(num_actions):
                strategy[i] = max(0, regrets.get(i, 0)) / sum_positive
        else:
            # Uniform strategy if no positive regrets
            for i in range(num_actions):
                strategy[i] = 1.0 / num_actions

        return strategy

    def evaluate_node(self, node_id, reach_probs_p0, reach_probs_p1, update_regrets):
        """Recursively evaluate a node."""
        node = self.nodes[node_id]

        if node['kind'] == 'chance':
            value = 0
            for outcome in node['outcomes']:
                prob = self.parse_number(outcome['prob'])
                child_id = outcome['child']
                child_value = self.evaluate_node(child_id, reach_probs_p0 * prob, reach_probs_p1 * prob, update_regrets)
                value += prob * child_value
            return value

        elif node['kind'] == 'decision':
            player = node['player']
            infoset = node['infoset']
            actions = node['actions']

            # Get strategy for this information set
            strategy = self.get_strategy(infoset, len(actions))

            # Determine whose reach probability to use
            my_reach = reach_probs_p0 if player == 0 else reach_probs_p1
            opp_reach = reach_probs_p1 if player == 0 else reach_probs_p0

            # Evaluate each action
            action_values = {}
            node_value = 0

            for i, action in enumerate(actions):
                child_id = action['child']
                if player == 0:
                    child_reach_p0 = reach_probs_p0 * strategy[i]
                    child_reach_p1 = reach_probs_p1
                else:
                    child_reach_p0 = reach_probs_p0
                    child_reach_p1 = reach_probs_p1 * strategy[i]

                child_value = self.evaluate_node(child_id, child_reach_p0, child_reach_p1, update_regrets)
                action_values[i] = child_value
                node_value += strategy[i] * child_value

            # Update regrets if requested
            if update_regrets:
                for i in range(len(actions)):
                    regret = action_values[i] - node_value
                    self.cumulative_regrets[infoset][i] += regret * opp_reach

            return node_value

        elif node['kind'] == 'terminal':
            return self.parse_number(node['payoff_p0'])

    def run_iteration(self):
        """Run one iteration of CFR."""
        self.evaluate_node(self.root, 1.0, 1.0, update_regrets=True)

    def get_average_strategy(self):
        """Compute the average strategy from regrets."""
        strategy_dict = {}

        # Collect all information sets
        infosets = set()
        for node in self.nodes.values():
            if node['kind'] == 'decision':
                infosets.add((node['infoset'], node['player'], len(node['actions'])))

        for infoset_name, player, num_actions in sorted(infosets):
            strategy = self.get_strategy(infoset_name, num_actions)
            strategy_dict[infoset_name] = strategy

        return strategy_dict

    def solve(self, num_iterations=10000):
        """Run CFR for a given number of iterations."""
        print(f"Running CFR for {num_iterations} iterations...")
        for i in range(num_iterations):
            if (i + 1) % 1000 == 0:
                print(f"  Iteration {i + 1}/{num_iterations}")
            self.run_iteration()
        print("Done!")

    def get_strategy_json(self):
        """Get the strategy in JSON format."""
        strategy = self.get_average_strategy()

        probs = {}
        for infoset_name in sorted(strategy.keys()):
            node_actions = None
            for node in self.nodes.values():
                if node['kind'] == 'decision' and node['infoset'] == infoset_name:
                    node_actions = node['actions']
                    break

            if node_actions:
                action_probs = {}
                strat = strategy[infoset_name]
                for i, action in enumerate(node_actions):
                    action_probs[action['label']] = str(strat[i])
                probs[infoset_name] = action_probs

        return {
            "format": "bluff-strategy-v1",
            "game": self.name,
            "producer": "CFR+ solver",
            "probs": probs
        }

def main():
    # Load game
    with open('game.json', 'r') as f:
        game_data = json.load(f)

    # Create and run solver
    solver = CFRSolver(game_data)
    solver.solve(num_iterations=10000)

    # Get strategy and write to file
    strategy_json = solver.get_strategy_json()
    with open('strategy.json', 'w') as f:
        json.dump(strategy_json, f, indent=2)

    print("Strategy written to strategy.json")

if __name__ == '__main__':
    main()
