#!/bin/bash

# Usage:
#  CONCURRENCY=8 ./docker-run-random.sh

set -euo pipefail

CONCURRENCY="${CONCURRENCY:-8}"
# Tend to use less CPU to generate more process context switches.
CPU="${CPU:-4}"
MEMORY="${MEMORY:-5g}"

TMP_PATH=""
TESTNAME="all"
TIDB_VERSION="v7.1.0"
REBUILD_IMAGE=1
RUN_ARGS=""

HELP=0

PWD=$(pwd)

while [[ $# -gt 0 ]]; do
	case "$1" in
	--tmp-path)
		TMP_PATH="$2"
		shift
		;;
	--test)
		TESTNAME="$2"
		shift
		;;
	--tidb-version)
		TIDB_VERSION="$2"
		shift
		;;
	--no-rebuild-image)
		REBUILD_IMAGE=0
		;;
	--keep-tmp-on-error)
		RUN_ARGS+=" --keep-tmp-on-error"
		;;
	--help)
		HELP=1
		break
		;;
	*)
		HELP=1
		break
		;;
	esac
	shift
done

if [ "$HELP" -eq 1 ]; then
	echo "Usage: $0 [OPTIONS]"
	echo "OPTIONS:"
	echo "  --help                             Display this message"
	echo "  --tmp-path     <temporary path>    Set the path for temporary data generated during testing"
	echo "  --test         <all/with_tidb>     Set the name of test case to run"
	echo "  --tidb-version <v6.6.0/v7.1.0/...> Set the version of TiDB for \"with_tidb\" test"
	echo "  --no-rebuild-image                 Do NOT rebuild the testing Docker image"
	echo "  --keep-tmp-on-error                Keep temporary data on error for debugging"
	exit 0
fi

BUILD_IMAGE_ARGS=""
if [ "$REBUILD_IMAGE" -eq 1 ]; then
	BUILD_IMAGE_ARGS+=" --pull --no-cache"
fi

IMAGE="ubuntu:20.04"
if [ "$TESTNAME" = "with_tidb" ]; then
	# Build image without context.
	docker build $BUILD_IMAGE_ARGS -t random-tidb --build-arg VERSION="$TIDB_VERSION" - < Dockerfile.tidb
	IMAGE="random-tidb"
fi

for((i=0;i<"$CONCURRENCY";i++)); do
  TMP_VOLUME=""
  if [ -n "$TMP_PATH" ]; then
    # Containers should use different $TMPDIR, otherwise they will conflict on TIKV_LOCK_FILES.
    mkdir -p "$TMP_PATH/$i"
    ABSOLUTE_TMP_PATH=$(readlink -f "$TMP_PATH/$i")
    TMP_VOLUME="-v $ABSOLUTE_TMP_PATH:/random-tmp"
  fi

  docker run --name random-"$TESTNAME"-"$i" --cpus="$CPU" --memory="$MEMORY" \
    -itd \
    -v "$PWD":/random \
    $TMP_VOLUME \
    "$IMAGE" \
    /bin/sh /random/run-random.sh "$i" "$TESTNAME" $RUN_ARGS
done
