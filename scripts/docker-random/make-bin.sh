#!/bin/bash

set -euo pipefail

PWD=$(pwd)
TARGET_BIN="$PWD"/random-bin

cd ../..

EXECUTABLE=$(cargo test -p tests --test random --no-run --message-format=json | grep -E -o 'target/debug/deps/random-[a-z0-9]+' | tail -1)
if [ -z "$EXECUTABLE" ]; then
    echo "Failed to find executable"
    exit 1
fi

# "rm" + "cp" to avoid target text file busy.
# "|| true" to avoid error if file does not exist.
rm "$TARGET_BIN" || true
cp "$EXECUTABLE" "$TARGET_BIN"
ls -l "$TARGET_BIN"
