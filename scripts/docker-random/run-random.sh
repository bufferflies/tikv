#!/bin/bash

set -eu

DOCKER_ID=$1
TESTNAME=$2

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
    else
        mv "$LOG" /random/error-logs/
    fi

    # TODO: optional keep data on error for debugging.
    # Note: When the container stops, the files from the last loop will not be automatically removed. However, they will be cleared during the next iterations.
    rm -rf "$TMPDIR" || true
done
