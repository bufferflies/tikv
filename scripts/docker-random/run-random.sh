#!/bin/bash

set -eu

DOCKER_ID=$1
TESTNAME=$2
shift 2

# Optional keep temporary data on error for debugging.
# Note: When the container stops, the files from the last loop will not be automatically removed. However, they will be cleared during the next iterations.
KEEP_TMP_ON_ERROR=0

while [ $# -gt 0 ]; do
    case "$1" in
        --keep-tmp-on-error)
            KEEP_TMP_ON_ERROR=1
            ;;
        *)
            echo "Usage: $0 DOCKER_ID TESTNAME [--keep-tmp-on-error]"
            exit 1
            ;;
    esac
    shift
done

export RUST_BACKTRACE=1

mkdir -p /random/logs /random/error-logs
for i in $(seq -w 1 100000)
do
    export TMPDIR="/random-tmp/$i"
    mkdir -p "$TMPDIR"

    LOG=/random/logs/random_"$TESTNAME"_"$i"_"$DOCKER_ID".log
    /random/random-bin test_random_"$TESTNAME" > "$LOG" 2>&1 || true
    if grep -q 'TEST SUCCEED' "$LOG"; then
        grep 'TEST SUCCEED' "$LOG"
        rm "$LOG"
        rm -rf "$TMPDIR" || true
    else
        mv "$LOG" /random/error-logs/
        if [ "$KEEP_TMP_ON_ERROR" -ne 1 ]; then
            rm -rf "$TMPDIR" || true
        fi
    fi

    if [ "$TESTNAME" = "with_tidb" ]; then
        pkill -9 tidb-server || true
        pkill -9 pd-server || true
    fi
done
