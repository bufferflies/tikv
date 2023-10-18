#!/bin/bash

set -euo pipefail

RELEASE=1
HELP=0
while [[ $# -gt 0 ]]; do
	case "$1" in
	--debug)
		RELEASE=0
		shift
		;;
	--help)
		HELP=1
		shift
		;;
	*)
		HELP=1
		break
		;;
	esac
done

if [ "$HELP" -eq 1 ]; then
	echo "Usage: $0 [OPTIONS]"
	echo "OPTIONS:"
	echo "  --help     Display this message"
	echo "  --debug    Make test binary as debug target"
	exit 0
fi

PWD=$(pwd)
TARGET_BIN="$PWD"/random-bin

BUILD_FLAG=""
TARGET_PATH="target/debug/deps"
if [ "$RELEASE" -eq 1 ]; then
    BUILD_FLAG="$BUILD_FLAG --release"
    TARGET_PATH="target/release/deps"
fi

cd ../..

EXECUTABLE=$(cargo test $BUILD_FLAG -p tests --test random --no-run --message-format=json | grep -E -o "$TARGET_PATH"'/random-[a-z0-9]+' | tail -1)
if [ -z "$EXECUTABLE" ]; then
    echo "Failed to find executable"
    exit 1
fi

# "rm" + "cp" to avoid target text file busy.
# "|| true" to avoid error if file does not exist.
rm "$TARGET_BIN" || true
cp "$EXECUTABLE" "$TARGET_BIN"
ls -l "$TARGET_BIN"
