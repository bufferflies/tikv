#!/bin/bash

set -euo pipefail

# Run from the Makefile environment
SELF=$(realpath -s "$0")
MAKEFILE_RUN=${MAKEFILE_RUN:-""}
if [[ -z $MAKEFILE_RUN ]]; then
	COMMAND="$SELF $*" exec make -f "$(dirname "$0")/../../Makefile" run
fi

show_help() {
	echo "Usage: $0 [OPTIONS]"
	echo "OPTIONS:"
	echo "  --help        Display this message"
	echo "  --debug       Make test binary as debug target"
	echo "  --env-logger  Enable env-logger"
}

RELEASE=1
ENV_LOGGER=0

while [[ $# -gt 0 ]]; do
	case "$1" in
	--debug)
		RELEASE=0
		shift
		;;
	--env-logger)
		ENV_LOGGER=1
		shift
		;;
	--help)
		show_help
		exit 0
		;;
	*)
		show_help
		exit 1
		;;
	esac
done

if [ "$ENV_LOGGER" -eq 1 ]; then
	TIKV_ENABLE_FEATURES="$TIKV_ENABLE_FEATURES env-logger"
fi

declare -a BUILD_FLAG
BUILD_FLAG=(
	"--features"
	"$TIKV_ENABLE_FEATURES"
)

PWD=$(pwd)
TARGET_BIN="$PWD"/random-bin

TARGET_PATH="target/debug/deps"
if [ "$RELEASE" -eq 1 ]; then
	BUILD_FLAG+=("--profile" "random-test")
	TARGET_PATH="target/random-test/deps"
fi

cd ../..

# "-p tikv" must be added, otherwise building will fail as some features are not contained in any package.
EXECUTABLE=$(cargo test "${BUILD_FLAG[@]}" -p tests -p tikv --test random --no-run --message-format=json | grep -E -o "$TARGET_PATH"'/random-[a-z0-9]+' | tail -1)
if [ -z "$EXECUTABLE" ]; then
	echo "Failed to find executable"
	exit 1
fi

# "rm" + "cp" to avoid target text file busy.
# "|| true" to avoid error if file does not exist.
rm "$TARGET_BIN" || true
cp "$EXECUTABLE" "$TARGET_BIN"
ls -l "$TARGET_BIN"
