#!/bin/sh
# Audit, verify and compile attempt C's core to Python (build/core-py).
set -e
cd "$(dirname "$0")"
SRC="Model.dfy SeqForm.dfy Proofs.dfy Recall.dfy Check.dfy"
if grep -nE 'assume|\{:axiom\}|\{:verify false\}|\{:extern' $SRC; then
  echo "audit: forbidden construct in the core" >&2
  exit 1
fi
echo "audit: no assume, {:axiom}, {:verify false} or {:extern} in $SRC"
dafny verify $SRC --cores 2
rm -rf build
dafny build --target:py --no-verify Check.dfy -o build/core
echo "built build/core-py"
