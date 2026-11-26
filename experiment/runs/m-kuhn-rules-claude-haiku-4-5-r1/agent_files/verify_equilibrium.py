from kuhn_poker import KuhnPoker
import json

poker = KuhnPoker()
poker.solve(iterations=100000)

avg_strategy = poker.get_average_strategy()

# Load and display the strategy
print("Final Strategy (100k iterations):")
for info_set in sorted(avg_strategy.keys()):
    print(f"{info_set}: {avg_strategy[info_set]}")

game_value = poker.compute_game_value(avg_strategy)
print(f"\nGame value: {game_value}")

# Verify by checking the regret values - they should be small at convergence
print("\nRegret values (should converge to 0):")
for info_set in sorted(poker.regret_sums.keys()):
    regrets = poker.regret_sums[info_set]
    print(f"{info_set}: {regrets}")

# Save the strategy
poker.save_strategy()
