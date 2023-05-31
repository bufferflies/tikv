#!/bin/bash

set -eu

NAME="all"

mkdir -p /random/logs /random/error-logs
for i in $(seq 1 100000)
do
    LOG=/random/logs/random_"$NAME"_$1_$i.log
    /random/random-bin test_random_"$NAME" > "$LOG" 2>&1 || true
    if grep -q 'TEST SUCCEED' "$LOG"; then
        grep 'TEST SUCCEED' "$LOG"
        rm "$LOG"
    else
        mv "$LOG" /random/error-logs/
    fi
done
