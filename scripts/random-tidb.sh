#!/usr/bin/env bash

export RUST_BACKTRACE=1

mkdir -p random_logs

for i in {1..1000}
do
    pkill -9 pd-server || true
    pkill -9 tidb-server || true

    TMPDIR=$(pwd)"/random_tmp/$i"
    export TMPDIR
    mkdir -p  "$TMPDIR"

    cargo test -p tests --test random test_random_with_tidb > random_logs/random_tidb_"$i".log 2>&1
    if grep -q 'TEST SUCCEED' random_logs/random_tidb_"$i".log; then
        echo -n "$i "
        grep 'TEST SUCCEED' random_logs/random_tidb_"$i".log;
        rm random_logs/random_tidb_"$i".log
    fi

    rm -rf "$TMPDIR" || true
done
