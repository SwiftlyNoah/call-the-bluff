"""One-shot simultaneous-move matrix games written as extensive-form trees.

KIND: normal-form (matrix) games, simultaneous move. Player 0 moves first in the tree; player 1's
nodes all share ONE infoset, so player 1 chooses without seeing player 0's choice. That single
infoset is what makes the move simultaneous.

Games and known values (to player 0):
  matching_pennies   H/T, +1 to p0 on a match, -1 otherwise.          value 0
  rps                rock-paper-scissors, +1 win, -1 loss, 0 tie.      value 0
  rps_biased         RPS where p0 winning WITH ROCK pays 2 (only p0).  value 1/12
                     Derivation in games/other/README.md: the unique equilibrium is
                     p0 = (R 1/4, P 5/12, S 1/3), p1 = (R 1/3, P 5/12, S 1/4).
"""

from __future__ import annotations

from fractions import Fraction

from gamelib import Tree, game_doc

PENNIES = (["H", "T"], [[1, -1], [-1, 1]])
RPS = (["R", "P", "S"], [[0, -1, 1], [1, 0, -1], [-1, 1, 0]])
RPS_BIASED = (["R", "P", "S"], [[0, -1, 2], [1, 0, -1], [-1, 1, 0]])

P0_INFOSET = "P0"
P1_INFOSET = "P1"


def matrix_tree(labels: list[str], payoff: list[list[int]]) -> Tree:
    t = Tree()
    t.decision(
        0,
        P0_INFOSET,
        [
            (
                a,
                lambda i=i: t.decision(
                    1,
                    P1_INFOSET,
                    [(b, lambda i=i, j=j: t.terminal(payoff[i][j])) for j, b in enumerate(labels)],
                ),
            )
            for i, a in enumerate(labels)
        ],
    )
    return t


def matching_pennies() -> dict:
    return game_doc(
        "matching_pennies",
        "Matching pennies as a tree: p0 picks H or T, p1 picks H or T in one infoset without seeing "
        "p0's choice (simultaneous move). p0 wins 1 on a match, loses 1 otherwise.",
        matrix_tree(*PENNIES),
        known_value=0,
        source="symmetric zero-sum game; equilibrium is uniform for both players (von Neumann)",
    )


def rps() -> dict:
    return game_doc(
        "rps",
        "Rock-paper-scissors as a tree: p0 moves, p1 moves in one infoset without seeing p0's "
        "choice (simultaneous move). Win +1, loss -1, tie 0.",
        matrix_tree(*RPS),
        known_value=0,
        source="antisymmetric payoff matrix, so the value is 0; uniform play is the equilibrium",
    )


def rps_biased() -> dict:
    return game_doc(
        "rps_biased",
        "Biased rock-paper-scissors as a tree (simultaneous move via one p1 infoset). As RPS, except "
        "that p0 winning with rock (R vs S) pays 2 to p0. p1 winning with rock still pays 1.",
        matrix_tree(*RPS_BIASED),
        known_value=Fraction(1, 12),
        source="closed form by indifference, derived in games/other/README.md: "
        "p0 (1/4, 5/12, 1/3), p1 (1/3, 5/12, 1/4) over (R, P, S)",
    )


# The hand-derived equilibrium, used by the tests to certify the value with exact best response.
RPS_BIASED_EQ = {
    P0_INFOSET: {"R": Fraction(1, 4), "P": Fraction(5, 12), "S": Fraction(1, 3)},
    P1_INFOSET: {"R": Fraction(1, 3), "P": Fraction(5, 12), "S": Fraction(1, 4)},
}

GAMES = [matching_pennies, rps, rps_biased]
