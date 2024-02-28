#!/usr/bin/env bash

set -euo pipefail

# Run from the Makefile environment
SELF=$(realpath -s "$0")
MAKEFILE_RUN=${MAKEFILE_RUN:-""}
if [[ -z $MAKEFILE_RUN ]]; then
    COMMAND="$SELF $*" exec make -f "$(dirname "$0")/../Makefile" run
fi

cargo test -p tests --test random --no-run

mkdir -p random_logs

for i in {1..1000}; do
    TMPDIR=$(pwd)"/random_tmp/$i"
    export TMPDIR
    mkdir -p "$TMPDIR"

    cargo test -p tests --test random test_random_all >random_logs/random_all_"$i".log 2>&1
    if grep -q 'TEST SUCCEED' random_logs/random_all_"$i".log; then
        echo -n "$i "
        grep 'TEST SUCCEED' random_logs/random_all_"$i".log
        rm random_logs/random_all_"$i".log
    fi

    rm -rf "$TMPDIR" || true
done
