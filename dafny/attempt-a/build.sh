#!/usr/bin/env bash
# Verify the attempt-a core, audit it, and compile it to Python for check.py.
set -euo pipefail
cd "$(dirname "$0")"
if grep -nE '\bassume\b|\{:axiom\}|\{:verify +false\}|\{:extern' Math.dfy Game.dfy BestResponse.dfy Check.dfy; then
  echo "audit: forbidden construct in the core" >&2
  exit 1
fi
dafny verify --verify-included-files Check.dfy
rm -rf build/checker-py
dafny build --target:py --no-verify Check.dfy --output build/checker
