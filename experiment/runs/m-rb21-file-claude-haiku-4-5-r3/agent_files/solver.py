#!/usr/bin/env python3
"""CFR+ solver for mini Red/Black game."""

import json
from fractions import Fraction
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional


def parse_number(s: str) -> Fraction:
    """Parse a number string (integer, fraction, or decimal) to Fraction."""
    if '/' in s:
        return Fraction(s)
    elif '.' in s:
        return Fraction(s).limit_denominator()
    else:
        return Fraction(int(s))


class GameNode:
    """Represents a node in the game tree."""
    def __init__(self, node_id: int, kind: str):
        self.id = node_id
        self.kind = kind


class ChanceNode(GameNode):
    """A chance node."""
    def __init__(self, node_id: int, outcomes: List[Tuple[str, Fraction, int]]):
        super().__init__(node_id, 'chance')
        self.outcomes = outcomes  # List of (label, prob, child_id)


class DecisionNode(GameNode):
    """A decision node."""
    def __init__(self, node_id: int, player: int, infoset: str, actions: List[Tuple[str, int]]):
        super().__init__(node_id, 'decision')
        self.player = player
        self.infoset = infoset
        self.actions = actions  # List of (label, child_id)


class TerminalNode(GameNode):
    """A terminal node."""
    def __init__(self, node_id: int, payoff_p0: Fraction):
        super().__init__(node_id, 'terminal')
        self.payoff_p0 = payoff_p0


class GameTree:
    """Represents the entire game tree."""
    def __init__(self, filename: str):
        with open(filename) as f:
            data = json.load(f)

        self.name = data.get('name', 'game')
        self.root_id = data['root']
        self.nodes: Dict[int, GameNode] = {}

        for node_data in data['nodes']:
            node_id = node_data['id']
            kind = node_data['kind']

            if kind == 'chance':
                outcomes = [
                    (o['label'], parse_number(o['prob']), o['child'])
                    for o in node_data['outcomes']
                ]
                self.nodes[node_id] = ChanceNode(node_id, outcomes)
            elif kind == 'decision':
                actions = [
                    (a['label'], a['child'])
                    for a in node_data['actions']
                ]
                self.nodes[node_id] = DecisionNode(
                    node_id, node_data['player'], node_data['infoset'], actions
                )
            elif kind == 'terminal':
                self.nodes[node_id] = TerminalNode(
                    node_id, parse_number(node_data['payoff_p0'])
                )

    def get_node(self, node_id: int) -> GameNode:
        return self.nodes[node_id]


class CFRSolver:
    """CFR+ solver."""

    def __init__(self, game: GameTree):
        self.game = game
        self.infoset_actions: Dict[str, List[str]] = defaultdict(list)
        self.cumulative_regrets: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.cumulative_strategy: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.iterations = 0
        self.regret_history: List[float] = []
        self.iteration_regrets: List[float] = []

        # Build infoset action mapping
        for node in game.nodes.values():
            if isinstance(node, DecisionNode):
                if node.infoset not in self.infoset_actions:
                    self.infoset_actions[node.infoset] = [action for action, _ in node.actions]

    def run(self, num_iterations: int = 1000):
        """Run CFR+ for num_iterations."""
        for t in range(num_iterations):
            self.iterations = t + 1
            self._run_iteration(self.game.root_id, {0: Fraction(1), 1: Fraction(1)})

            # Track total regrets every 100 iterations
            if (t + 1) % 100 == 0:
                total_regret = sum(
                    abs(r) for regrets in self.cumulative_regrets.values()
                    for r in regrets.values()
                )
                self.regret_history.append(total_regret)

    def _run_iteration(self, node_id: int, reach_probs: Dict[int, Fraction]) -> Fraction:
        """Run one iteration of CFR and return expected payoff for player 0."""
        node = self.game.get_node(node_id)

        if isinstance(node, TerminalNode):
            return node.payoff_p0

        if isinstance(node, ChanceNode):
            value = Fraction(0)
            for label, prob, child_id in node.outcomes:
                new_reach = reach_probs.copy()
                # Both players see chance
                new_reach[0] *= prob
                new_reach[1] *= prob
                child_value = self._run_iteration(child_id, new_reach)
                value += prob * child_value
            return value

        # DecisionNode
        player = node.player
        infoset = node.infoset
        actions = node.actions

        # Get strategy for this infoset
        strategy = self._get_strategy(infoset)

        # Compute value of each action
        action_values = []
        for action_label, child_id in actions:
            new_reach = reach_probs.copy()
            new_reach[player] *= Fraction(strategy[action_label])
            child_value = self._run_iteration(child_id, new_reach)
            action_values.append(child_value)

        # Compute node value
        node_value = sum(
            Fraction(strategy[actions[i][0]]) * action_values[i]
            for i in range(len(actions))
        )

        # Update cumulative strategy
        for i, (action_label, _) in enumerate(actions):
            self.cumulative_strategy[infoset][action_label] += float(strategy[action_label])

        # Compute regrets and update cumulative regrets
        opp_reach = reach_probs[1 - player]
        for i, (action_label, _) in enumerate(actions):
            regret = float(opp_reach) * float(action_values[i] - node_value)
            self.cumulative_regrets[infoset][action_label] += regret

        return node_value

    def _get_strategy(self, infoset: str) -> Dict[str, float]:
        """Get current strategy for infoset using regret matching."""
        actions = self.infoset_actions[infoset]

        # Regret matching: use positive regrets
        positive_regrets = {}
        total_regret = 0.0
        for action in actions:
            regret = max(0.0, self.cumulative_regrets[infoset][action])
            positive_regrets[action] = regret
            total_regret += regret

        # Normalize to probabilities
        if total_regret > 0:
            strategy = {
                action: positive_regrets[action] / total_regret
                for action in actions
            }
        else:
            # Uniform strategy if no positive regrets
            strategy = {action: 1.0 / len(actions) for action in actions}

        return strategy

    def get_average_strategy(self) -> Dict[str, Dict[str, str]]:
        """Get average strategy as a dictionary."""
        result = {}
        for infoset, action_counts in self.cumulative_strategy.items():
            actions = self.infoset_actions[infoset]
            # Use Fraction throughout to avoid floating-point errors
            total = Fraction(0)
            counts_frac = {}
            for action in actions:
                count = Fraction(action_counts.get(action, 0)).limit_denominator(1000000)
                counts_frac[action] = count
                total += count

            if total > 0:
                probs = {}
                for action in actions:
                    prob_frac = counts_frac[action] / total
                    # Simplify and represent as string
                    probs[action] = str(prob_frac)
            else:
                probs = {action: str(Fraction(1, len(actions))) for action in actions}

            result[infoset] = probs

        return result

    def compute_game_value(self) -> float:
        """Compute expected value of the game under current average strategy."""
        avg_strat = {}
        for infoset, action_counts in self.cumulative_strategy.items():
            actions = self.infoset_actions[infoset]
            total = sum(action_counts.get(a, 0) for a in actions)
            if total > 0:
                avg_strat[infoset] = {
                    action: action_counts.get(action, 0) / total
                    for action in actions
                }
            else:
                avg_strat[infoset] = {action: 1.0 / len(actions) for action in actions}

        return float(self._eval_value(self.game.root_id, avg_strat, {0: 1.0, 1: 1.0}))

    def _eval_value(self, node_id: int, strategy: Dict[str, Dict[str, float]], reach: Dict[int, float]) -> Fraction:
        """Evaluate game value under given strategy."""
        node = self.game.get_node(node_id)

        if isinstance(node, TerminalNode):
            return node.payoff_p0

        if isinstance(node, ChanceNode):
            value = Fraction(0)
            for label, prob, child_id in node.outcomes:
                new_reach = reach.copy()
                new_reach[0] *= float(prob)
                new_reach[1] *= float(prob)
                child_value = self._eval_value(child_id, strategy, new_reach)
                value += prob * child_value
            return value

        # DecisionNode
        player = node.player
        infoset = node.infoset
        actions = node.actions
        strat = strategy.get(infoset, {a: 1.0/len(actions) for a, _ in actions})

        value = Fraction(0)
        for action_label, child_id in actions:
            prob = Fraction(strat.get(action_label, 1.0 / len(actions)))
            new_reach = reach.copy()
            new_reach[player] *= float(prob)
            child_value = self._eval_value(child_id, strategy, new_reach)
            value += prob * child_value

        return value

    def get_metrics(self) -> Dict[str, Any]:
        """Get convergence metrics."""
        return {
            'iterations': self.iterations,
            'regret_history': self.regret_history,
        }


def main():
    game = GameTree('game.json')
    solver = CFRSolver(game)
    print("Running CFR+ solver...")
    solver.run(num_iterations=5000)
    print(f"Completed {solver.iterations} iterations")

    # Compute metrics
    game_value = solver.compute_game_value()
    print(f"Game value for player 0: {game_value:.6f}")

    metrics = solver.get_metrics()
    if metrics['regret_history']:
        print(f"Initial total regret (iter 100): {metrics['regret_history'][0]:.2f}")
        print(f"Final total regret (iter 2000): {metrics['regret_history'][-1]:.2f}")
        convergence_rate = metrics['regret_history'][0] / max(metrics['regret_history'][-1], 1e-10)
        print(f"Convergence ratio (initial/final): {convergence_rate:.2f}x")

    avg_strategy = solver.get_average_strategy()

    output = {
        'format': 'bluff-strategy-v1',
        'game': game.name,
        'producer': 'CFR+ solver',
        'probs': avg_strategy
    }

    with open('strategy.json', 'w') as f:
        json.dump(output, f, indent=2)

    print("Strategy written to strategy.json")


if __name__ == '__main__':
    main()
