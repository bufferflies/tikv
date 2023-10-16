#!/bin/bash

set -euo pipefail

CONCURRENCY="${CONCURRENCY:-8}"

stop() {
  docker stop random-all-$1 && docker rm random-all-$1
}

for ((i=0;i<"$CONCURRENCY";i++)); do
  stop "$i" &
done

wait
