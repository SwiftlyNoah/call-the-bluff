#!/bin/sh
# Verify the core, then compile it to Python under build/bluffcore-py.
set -e
cd "$(dirname "$0")"
dafny verify Model.dfy BestResponse.dfy Check.dfy --cores 2
dafny build --target:py Check.dfy --no-verify -o build/bluffcore
