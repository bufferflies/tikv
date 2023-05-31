#!/bin/bash

set -euo pipefail

CONCURRENCY="${CONCURRENCY:-8}"

for ((i=0;i<"$CONCURRENCY";i++)); do
  docker stop random-all-$i && docker rm random-all-$i
done
