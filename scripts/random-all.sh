#!/usr/bin/env bash

TMPDIR=$(pwd)"/random_tmp"
export TMPDIR

mkdir -p random_logs "$TMPDIR"

for i in {1..1000}
do
    cargo test -p tests --test random test_random_all > random_logs/random_all_"$i".log 2>&1
    if grep -q 'TEST SUCCEED' random_logs/random_all_"$i".log; then
        echo -n "$i "
        grep 'TEST SUCCEED' random_logs/random_all_"$i".log;
        rm random_logs/random_all_"$i".log
    fi
done
