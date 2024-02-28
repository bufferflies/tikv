#!/bin/bash

# To daily run:
# crontab -e
# 0 0 * * * /xxx/cloud-storage-engine/scripts/docker-random/tidb.sh >> /tmp/random-tidb.log 2>&1

set -xeuo pipefail
CONCURRENCY=4
CPU=4
MEMORY=8g

WORKDIR="/data/nvme1n1/$LOGNAME/random"

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
    --cpu)
        CPU="$2"
        shift
        ;;
    --memory)
        MEMORY="$2"
        shift
        ;;
    *)
        echo "Usage: $0 --work-dir <WORKDIR> --concurrency <CONCURRENCY> --cpu <CPU> --memory <MEMORY>"
        exit 1
        ;;
    esac
    shift
done

echo "================== Radnom Test for TiDB $(date) =================="

source "$HOME/.cargo/env"

CWD=$(dirname "$(realpath -s "$0")")
cd "$CWD" || exit 1

git pull
git merge origin/cloud-engine --signoff --no-edit
./make-bin.sh

mkdir -p "$WORKDIR"

export CONCURRENCY
export CPU
export MEMORY

./docker-stop-random.sh --test with_tidb
./docker-run-random.sh \
    --path-with-suffix \
    --keep-tmp-on-error \
    --test with_tidb \
    --log-path "$WORKDIR/tidb-logs" \
    --tmp-path "$WORKDIR/tidb-tmp"
