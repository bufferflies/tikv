#!/bin/bash

# Usage:
#  CONCURRENCY=8 ./docker-run-random.sh

set -euo pipefail

CONCURRENCY="${CONCURRENCY:-8}"
# Tend to use less CPU to generate more process context switches.
CPU="${CPU:-4}"
MEMORY="${MEMORY:-5g}"
TESTNAME="${TESTNAME:-all}"
PWD=$(pwd)
TMP_PATH=""
HELP=0

while [[ $# -gt 0 ]]; do
	case "$1" in
	--tmp-path)
		TMP_PATH="$2"
		shift
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
	echo "  --help                       Display this message"
	echo "  --tmp-path <temporary path>  Set the path for temporary data generated during testing"
	exit 0
fi

for((i=0;i<"$CONCURRENCY";i++)); do
  TMP_VOLUME=""
  if [ -n "$TMP_PATH" ]; then
    # Containers should use different $TMPDIR, otherwise they will conflict on TIKV_LOCK_FILES.
    mkdir -p "$TMP_PATH/$i"
    ABSOLUTE_TMP_PATH=$(readlink -f "$TMP_PATH/$i")
    TMP_VOLUME="-v $ABSOLUTE_TMP_PATH:/random-tmp"
  fi

  docker run --name random-all-"$i" --cpus="$CPU" --memory="$MEMORY" \
    -itd \
    -v "$PWD":/random \
    $TMP_VOLUME \
    ubuntu:20.04 \
    /bin/sh /random/run-random.sh "$i" "$TESTNAME"
done
