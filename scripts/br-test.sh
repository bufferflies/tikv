#!/usr/bin/env bash

for i in {1..1000}
do
    cargo test -p tests --test random random_br > random_br_$i.log 2>&1
    if grep -q 'total_write_count' random_br_$i.log; then
        grep 'total_write_count' random_br_$i.log;
        rm random_br_$i.log
    fi
done
