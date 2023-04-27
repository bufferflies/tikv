#!/usr/bin/env bash

mkdir -p random_br_logs

for i in {1..1000}
do
    cargo test -p tests --test random random_br > random_br_logs/random_br_$i.log 2>&1
    if grep -q 'TEST SUCCEED' random_br_logs/random_br_$i.log; then
        echo -n "$i "
        grep 'TEST SUCCEED' random_br_logs/random_br_$i.log;
        rm random_br_logs/random_br_$i.log
    fi
done
