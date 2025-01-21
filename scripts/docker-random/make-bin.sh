#!/bin/bash

set -eu

# Run from the Makefile environment
SELF=$(realpath -s "$0")
MAKEFILE_RUN=${MAKEFILE_RUN:-""}
if [[ -z $MAKEFILE_RUN ]]; then
	COMMAND="$SELF $*" exec make -f "$(dirname "$0")/../../Makefile" run
fi

show_help() {
	echo "Usage: $0 [OPTIONS]"
	echo "OPTIONS:"
	echo "  --help             Display this message"
	echo "  --debug            Make test binary as debug target"
	echo "  --env-logger       Enable env-logger"
	echo "  --debug-trace-xxx  Enable debug trace"
}

RELEASE=1
EXTRA_FEATURES=()

while [[ $# -gt 0 ]]; do
	case "$1" in
	--debug)
		RELEASE=0
		;;
	--env-logger)
		EXTRA_FEATURES+=("env-logger")
		;;
	--debug-trace-mem-table)
		EXTRA_FEATURES+=("debug-trace-mem-table")
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
	shift
done

# Concat by space
EXTRA_FEATURES_STR=$(IFS=$' '; echo "${EXTRA_FEATURES[*]}")
TIKV_ENABLE_FEATURES="$TIKV_ENABLE_FEATURES $EXTRA_FEATURES_STR"
echo "features: $TIKV_ENABLE_FEATURES"

declare -a BUILD_FLAG
BUILD_FLAG=(
	"--features"
	"$TIKV_ENABLE_FEATURES"
)

PWD=$(pwd)
TARGET_BIN="$PWD"/random-bin
BUILD_LOG="$PWD"/build.log

TARGET_PATH="target/debug/deps"
if [ "$RELEASE" -eq 1 ]; then
	BUILD_FLAG+=("--profile" "random-test")
	TARGET_PATH="target/random-test/deps"
fi

cd ../..

# "-p tikv" must be added, otherwise building will fail as some features are not contained in any package.
EXECUTABLE=$(cargo test "${BUILD_FLAG[@]}" -p tests -p tikv --test random --no-run --message-format=json | tee "$BUILD_LOG" | grep -E -o "$TARGET_PATH"'/random-[a-z0-9]+' | tail -1)
if [ -z "$EXECUTABLE" ]; then
	echo "Failed to find executable, errors:"
	echo
	jq -r 'select(.reason == "compiler-message" and .message.level == "error") | .message.message' < "$BUILD_LOG"
	echo
	echo "See build.log for details."
	exit 1
fi

# "rm" + "cp" to avoid target text file busy.
# "|| true" to avoid error if file does not exist.
rm "$TARGET_BIN" || true
cp "$EXECUTABLE" "$TARGET_BIN"
ls -l "$TARGET_BIN"
