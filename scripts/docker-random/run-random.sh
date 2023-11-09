#!/bin/bash

set -eu

DOCKER_ID=$1
TESTNAME=$2

TMPDIR="/random-tmp"
export TMPDIR

mkdir -p /random/logs /random/error-logs "$TMPDIR"
for i in $(seq -w 1 100000)
do
    LOG=/random/logs/random_"$TESTNAME"_"$i"_"$DOCKER_ID".log
    /random/random-bin test_random_"$TESTNAME" > "$LOG" 2>&1 || true
    if grep -q 'TEST SUCCEED' "$LOG"; then
        grep 'TEST SUCCEED' "$LOG"
        rm "$LOG"
    else
        mv "$LOG" /random/error-logs/
    fi
done
