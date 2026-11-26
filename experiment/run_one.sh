#!/bin/bash
# Run one agent trial and grade it. Arguments pass through to `run.py trial`, e.g.
#   experiment/run_one.sh --game kuhn --variant file --model claude-haiku-5-5
set -euo pipefail
cd "$(dirname "$0")/.."
exec uv run experiment/run.py trial "$@"
