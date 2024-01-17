#!/usr/bin/env bash

KEEP_TMP_ON_ERROR=0

while [ $# -gt 0 ]; do
    case "$1" in
        --keep-tmp-on-error)
            KEEP_TMP_ON_ERROR=1
            ;;
        *)
            echo "Usage: $0 [--keep-tmp-on-error]"
            exit 1
            ;;
    esac
    shift
done

export RUST_BACKTRACE=1

mkdir -p random_logs

for i in {1..1000}
do
    pkill -9 pd-server &>/dev/null || true
    pkill -9 tidb-server &>/dev/null || true

    TMPDIR=$(pwd)"/random_tmp/$i"
    export TMPDIR
    mkdir -p  "$TMPDIR"

    cargo test -p tests --test random test_random_with_tidb > random_logs/random_tidb_"$i".log 2>&1
    if grep -q 'TEST SUCCEED' random_logs/random_tidb_"$i".log; then
        echo -n "$i "
        grep 'TEST SUCCEED' random_logs/random_tidb_"$i".log;
        rm random_logs/random_tidb_"$i".log
        rm -rf "$TMPDIR" || true
    else
        if [ "$KEEP_TMP_ON_ERROR" -ne 1 ]; then
            rm -rf "$TMPDIR" || true
        fi
    fi

done
