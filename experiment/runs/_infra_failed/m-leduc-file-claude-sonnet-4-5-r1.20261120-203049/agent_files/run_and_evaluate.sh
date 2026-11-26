#!/bin/bash
# Run CFR solver and evaluate the result

echo "Running CFR+ solver..."
python3 cfr_solver.py

if [ -f strategy.json ]; then
    echo ""
    echo "Strategy generated. Evaluating..."
    python3 evaluate_strategy.py
    echo ""
    echo "Complete! Results:"
    cat evaluation.json
else
    echo "Error: strategy.json not generated"
    exit 1
fi
