#!/bin/bash

# To daily run:
# crontab -e
# 0 0 * * * /xxx/cloud-storage-engine/scripts/docker-random/all.sh >> /tmp/random-all.log 2>&1

set -xeuo pipefail

WORKDIR="/data/nvme1n1/$LOGNAME/random"
CONCURRENCY=12
CPU=3
MEMORY=3g
MEMORY_PROFILE=0

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
    --memory-profile)
        MEMORY_PROFILE=1
        ;;
    *)
        echo "Usage: $0 --work-dir <WORKDIR> --concurrency <CONCURRENCY> --cpu <CPU> --memory <MEMORY> --memory-profile"
        exit 1
        ;;
    esac
    shift
done

echo "================== Radnom Test for All $(date) =================="

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

declare -a RUN_ARGS
RUN_ARGS=(
    "--path-with-suffix"
    "--log-path" "$WORKDIR/all-logs"
)

if [ "$MEMORY_PROFILE" -eq 1 ]; then
    RUN_ARGS+=(
        "--memory-profile"
        "--tmp-path" "$WORKDIR/all-tmp"
        "--keep-tmp-on-error"
    )
fi

./docker-stop-random.sh
./docker-run-random.sh "${RUN_ARGS[@]}"
