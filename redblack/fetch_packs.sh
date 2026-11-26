#!/usr/bin/env bash
# Fetch the served Red/Black pack and values side-car for every class with
# post-discard total T' <= 3 into redblack/out/{pack,values}/. READ ONLY on
# the bucket: only `gcloud storage cp` from gs:// to local.
#
#   RB_GCLOUD_ACCOUNT=<an account with read access to red-black-dev> redblack/fetch_packs.sh
#
# The default gcloud account on this Mac gets a 403 on the bucket; pass one
# that can read it. The class list is redblack/out/classes_le3.txt (written by
# `rounds.feasible_classes(3)`); the 45 `dr-` ids in it are not in the pack
# (the only served root is dr-5.0.0-5.0.0) and fail to copy, which is expected.
set -euo pipefail
cd "$(dirname "$0")/out"
BUCKET=gs://red-black-dev-solver
PACK=solver-pack-v2-profile-solve9b
VALUES=solver-values-v1-solve9b
export CLOUDSDK_CORE_ACCOUNT="${RB_GCLOUD_ACCOUNT:?set RB_GCLOUD_ACCOUNT}"
mkdir -p pack values
grep -v '^dr-' classes_le3.txt | sed "s#.*#$BUCKET/$PACK/&.pack.gz#" | gcloud storage cp -I pack/
grep -v '^dr-' classes_le3.txt | sed "s#.*#$BUCKET/$VALUES/&.values.gz#" | gcloud storage cp -I values/
gcloud storage cp "$BUCKET/$PACK/index.json" pack/index.json
ls pack | wc -l
