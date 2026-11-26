import json
from kuhn_poker import KuhnPoker

poker = KuhnPoker()
print("Running CFR with convergence test...")

# Run for different numbers of iterations to see convergence
for iterations in [1000, 5000, 10000, 20000, 50000]:
    poker = KuhnPoker()
    poker.solve(iterations=iterations)
    avg_strategy = poker.get_average_strategy()
    game_value = poker.compute_game_value(avg_strategy)
    print(f"Iterations: {iterations:5d} -> Game value: {game_value:8.6f}")

