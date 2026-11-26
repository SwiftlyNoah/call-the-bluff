#!/usr/bin/env python3
"""
CFR solver for two-player zero-sum games in extensive form.
"""

import json
from collections import defaultdict
from typing import Dict


def parse_rational(s: str) -> float:
    """Parse a rational number string (integer, fraction, or decimal)."""
    if '/' in s:
        num, denom = s.split('/')
        return int(num) / int(denom)
    return float(s)


class GameTree:
    """Represents a game tree loaded from game.json."""
    
    def __init__(self, filename: str):
        with open(filename, 'r') as f:
            data = json.load(f)
        
        self.name = data['name']
        self.root = data['root']
        self.nodes = {node['id']: node for node in data['nodes']}
        
        # Build information set structure
        self.infosets = defaultdict(lambda: {'player': None, 'actions': []})
        for node in data['nodes']:
            if node['kind'] == 'decision':
                infoset = node['infoset']
                if self.infosets[infoset]['player'] is None:
                    self.infosets[infoset]['player'] = node['player']
                    self.infosets[infoset]['actions'] = [a['label'] for a in node['actions']]


class CFRSolver:
    """Counterfactual Regret Minimization solver."""
    
    def __init__(self, game: GameTree):
        self.game = game
        
        # Regret and strategy tracking for each information set
        self.regret_sum = defaultdict(lambda: defaultdict(float))
        self.strategy_sum = defaultdict(lambda: defaultdict(float))
        self.iterations = 0
    
    def get_strategy(self, infoset: str) -> Dict[str, float]:
        """Get current strategy for an information set using regret matching."""
        actions = self.game.infosets[infoset]['actions']
        strategy = {}
        
        # Sum positive regrets
        normalizing_sum = sum(max(0, self.regret_sum[infoset][action]) for action in actions)
        
        for action in actions:
            if normalizing_sum > 0:
                strategy[action] = max(0, self.regret_sum[infoset][action]) / normalizing_sum
            else:
                # Uniform distribution if no positive regrets
                strategy[action] = 1.0 / len(actions)
        
        return strategy
    
    def cfr(self, node_id: int, reach_prob_0: float, reach_prob_1: float) -> float:
        """
        Vanilla CFR algorithm.
        
        Returns the utility for player 0.
        """
        node = self.game.nodes[node_id]
        
        # Terminal node
        if node['kind'] == 'terminal':
            return parse_rational(node['payoff_p0'])
        
        # Chance node
        if node['kind'] == 'chance':
            utility = 0.0
            for outcome in node['outcomes']:
                prob = parse_rational(outcome['prob'])
                utility += prob * self.cfr(outcome['child'], reach_prob_0, reach_prob_1)
            return utility
        
        # Decision node
        player = node['player']
        infoset = node['infoset']
        
        # Get current strategy
        strategy = self.get_strategy(infoset)
        
        # Compute utility for each action
        util_action = {}
        for action_data in node['actions']:
            action = action_data['label']
            action_prob = strategy[action]
            
            # Recurse with updated reach probabilities
            if player == 0:
                util_action[action] = self.cfr(
                    action_data['child'],
                    reach_prob_0 * action_prob,
                    reach_prob_1
                )
            else:
                util_action[action] = self.cfr(
                    action_data['child'],
                    reach_prob_0,
                    reach_prob_1 * action_prob
                )
        
        # Utility of current strategy
        util_node = sum(strategy[action] * util_action[action] for action in strategy)
        
        # Update regret and strategy sum
        for action in strategy:
            # util_action and util_node are from player 0's perspective
            # Player 1 wants to minimize player 0's utility, so negate for player 1
            if player == 0:
                regret = util_action[action] - util_node
                self.regret_sum[infoset][action] += reach_prob_1 * regret
                self.strategy_sum[infoset][action] += reach_prob_0 * strategy[action]
            else:
                # For player 1, negate utilities to get player 1's perspective
                regret = util_node - util_action[action]
                self.regret_sum[infoset][action] += reach_prob_0 * regret
                self.strategy_sum[infoset][action] += reach_prob_1 * strategy[action]
        
        return util_node
    
    def train(self, iterations: int, print_every: int = 20000):
        """Run CFR for a specified number of iterations."""
        for i in range(iterations):
            self.cfr(self.game.root, 1.0, 1.0)
            self.iterations += 1
            
            if (i + 1) % print_every == 0:
                avg_strat = self.get_average_strategy()
                exploit = self.compute_exploitability(avg_strat)
                value = self.compute_strategy_value(avg_strat)
                print(f"Iter {i + 1}: exploitability={exploit:.6f}, value_p0={value:.6f}")
    
    def get_average_strategy(self) -> Dict[str, Dict[str, float]]:
        """Get the average strategy across all iterations."""
        avg_strategy = {}
        
        for infoset in self.game.infosets:
            actions = self.game.infosets[infoset]['actions']
            
            # Sum of strategy
            strategy_sum_total = sum(self.strategy_sum[infoset][action] for action in actions)
            
            avg_strategy[infoset] = {}
            for action in actions:
                if strategy_sum_total > 0:
                    avg_strategy[infoset][action] = self.strategy_sum[infoset][action] / strategy_sum_total
                else:
                    # If never reached, use uniform
                    avg_strategy[infoset][action] = 1.0 / len(actions)
        
        return avg_strategy
    
    def compute_exploitability(self, strategy: Dict[str, Dict[str, float]]) -> float:
        """
        Compute exploitability of a strategy.
        Exploitability = (BR_value_p0 + BR_value_p1) / 2
        """
        br_value_p0 = self._best_response_value(0, strategy)
        br_value_p1 = self._best_response_value(1, strategy)
        return (br_value_p0 + br_value_p1) / 2
    
    def _best_response_value(self, br_player: int, opp_strategy: Dict[str, Dict[str, float]]) -> float:
        """Compute best response value for a player against opponent's strategy."""
        
        def traverse_br(node_id: int) -> float:
            node = self.game.nodes[node_id]
            
            if node['kind'] == 'terminal':
                payoff = parse_rational(node['payoff_p0'])
                return payoff if br_player == 0 else -payoff
            
            if node['kind'] == 'chance':
                value = 0.0
                for outcome in node['outcomes']:
                    prob = parse_rational(outcome['prob'])
                    value += prob * traverse_br(outcome['child'])
                return value
            
            # Decision node
            player = node['player']
            infoset = node['infoset']
            
            if player == br_player:
                # Best response player: take maximum
                values = []
                for action_data in node['actions']:
                    values.append(traverse_br(action_data['child']))
                return max(values)
            else:
                # Opponent: use their strategy
                value = 0.0
                for action_data in node['actions']:
                    action = action_data['label']
                    prob = opp_strategy[infoset][action]
                    value += prob * traverse_br(action_data['child'])
                return value
        
        return traverse_br(self.game.root)
    
    def compute_strategy_value(self, strategy: Dict[str, Dict[str, float]]) -> float:
        """Compute expected value for player 0 when both players follow the strategy."""
        
        def traverse(node_id: int) -> float:
            node = self.game.nodes[node_id]
            
            if node['kind'] == 'terminal':
                return parse_rational(node['payoff_p0'])
            
            if node['kind'] == 'chance':
                value = 0.0
                for outcome in node['outcomes']:
                    prob = parse_rational(outcome['prob'])
                    value += prob * traverse(outcome['child'])
                return value
            
            # Decision node
            infoset = node['infoset']
            value = 0.0
            for action_data in node['actions']:
                action = action_data['label']
                prob = strategy[infoset][action]
                value += prob * traverse(action_data['child'])
            return value
        
        return traverse(self.game.root)


def main():
    # Load game
    print("Loading game...")
    game = GameTree('game.json')
    print(f"Game: {game.name}")
    print(f"Information sets: {len(game.infosets)}")
    
    # Create solver
    solver = CFRSolver(game)
    
    # Train
    print("\nTraining CFR...")
    num_iterations = 100000
    solver.train(num_iterations, print_every=20000)
    
    # Get average strategy
    avg_strategy = solver.get_average_strategy()
    
    # Compute metrics
    print("\nFinal results:")
    exploitability = solver.compute_exploitability(avg_strategy)
    value_p0 = solver.compute_strategy_value(avg_strategy)
    
    print(f"Exploitability: {exploitability:.8f}")
    print(f"Value for P0: {value_p0:.8f}")
    
    # Format strategy for output
    strategy_output = {
        "format": "bluff-strategy-v1",
        "game": game.name,
        "producer": f"CFR solver, {num_iterations} iterations",
        "probs": {}
    }
    
    for infoset, action_probs in avg_strategy.items():
        strategy_output["probs"][infoset] = {
            action: repr(prob) for action, prob in action_probs.items()
        }
    
    # Write strategy
    with open('strategy.json', 'w') as f:
        json.dump(strategy_output, f, indent=1)
    
    print("\nStrategy written to strategy.json")
    
    # Print strategy summary
    print("\nStrategy summary:")
    for infoset in sorted(avg_strategy.keys()):
        player = game.infosets[infoset]['player']
        print(f"  {infoset} (P{player}):", end="")
        for action, prob in avg_strategy[infoset].items():
            print(f" {action}={prob:.4f}", end="")
        print()


if __name__ == '__main__':
    main()
