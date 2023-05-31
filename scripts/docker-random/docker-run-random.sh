#!/bin/bash

# Usage:
#  CONCURRENCY=8 ./docker-run-random.sh

set -euo pipefail

CONCURRENCY="${CONCURRENCY:-8}"

# Tend to use less CPU to generate more process context switches.
CPU="${CPU:-4}"
MEMORY="${MEMORY:-5g}"

PWD=$(pwd)

for((i=0;i<"$CONCURRENCY";i++)); do
  docker run --name random-all-$i --cpus="$CPU" --memory="$MEMORY" -itd -v "$PWD":/random ubuntu:20.04 /bin/sh /random/run-random.sh $i
done
