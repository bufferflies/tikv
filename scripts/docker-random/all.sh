#!/bin/bash

# To daily run:
# crontab -e
# 0 0 * * * /xxx/cloud-storage-engine/scripts/docker-random/all.sh >> /tmp/random-all.log 2>&1

set -xeuo pipefail

WORKDIR="/data/nvme1n1/$LOGNAME/random"
CONCURRENCY=8

while [ $# -gt 0 ]; do
    case "$1" in
    --work-dir)
        WORKDIR="$2"
        shift
        ;;
    --concurrency)
        CONCURRENCY="$2"
        shift
        ;;
    *)
        echo "Usage: $0 --work-dir <WORKDIR> --concurrency <CONCURRENCY>"
        exit 1
        ;;
    esac
    shift
done

echo "================== Radnom Test for All $(date) =================="

source "$HOME/.cargo/env"

CWD=$(dirname "$(realpath -s "$0")")
cd "$CWD" || exit 1

./docker-stop-random.sh

git pull
git merge origin/cloud-engine --signoff
./make-bin.sh

mkdir -p "$WORKDIR"
export CONCURRENCY

./docker-run-random.sh \
    --path-with-suffix \
    --log-path "$WORKDIR/all-logs"
