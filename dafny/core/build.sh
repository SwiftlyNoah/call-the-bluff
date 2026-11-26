#!/usr/bin/env bash
# Verify the core and compile it to Python for adapter.py. bin/audit runs this after its
# forbidden-construct checks. The compiled core lands in build/checker-py together with
# SOURCE_SHA256, the hash of the .dfy files it was verified from, and BUILD_SHA256, the hash
# of every compiled file this run produced. adapter.py refuses to run unless both match, so
# no verdict comes from unverified sources or from compiled code edited after the build.
set -euo pipefail
cd "$(dirname "$0")"
SRC=(Math.dfy Game.dfy BestResponse.dfy Check.dfy)
hash() { uv run --quiet python -c 'import sys; sys.path.insert(0, "."); import adapter; print(adapter.source_hash())'; }
before=$(hash)
dafny verify "${SRC[@]}" --cores 2
mkdir -p build
tmp=$(mktemp -d build/.tmp.XXXXXX)
trap 'rm -rf "$tmp"' EXIT
dafny build --target:py --no-verify Check.dfy --output "$tmp/checker" >/dev/null
after=$(hash)
if [ "$before" != "$after" ]; then
  echo "build.sh: a .dfy file changed during the build; run it again" >&2
  exit 1
fi
echo "$after" > "$tmp/checker-py/SOURCE_SHA256"
uv run --quiet python -c 'import sys; sys.path.insert(0, "."); import adapter; from pathlib import Path; print(adapter.build_hash(Path(sys.argv[1])))' "$tmp/checker-py" > "$tmp/checker-py/BUILD_SHA256"
rm -rf build/checker-py.old
[ -d build/checker-py ] && mv build/checker-py build/checker-py.old
mv "$tmp/checker-py" build/checker-py
rm -rf build/checker-py.old
echo "build.sh: verified and compiled core at build/checker-py (sources $after)"
