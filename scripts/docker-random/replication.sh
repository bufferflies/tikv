#!/bin/bash

# To daily run:
# crontab -e
# 0 0 * * * /xxx/cloud-storage-engine/scripts/docker-random/replication.sh >> /tmp/random-rep.log 2>&1

set -euo pipefail

WORKDIR="/data/nvme1n1/$LOGNAME/random"
LOGDIR="/data/nvme0n1/$LOGNAME/random"
CONCURRENCY=4
CPU=4
MEMORY=8g
MEMORY_PROFILE=0
MAKE_BIN_ARGS=""
GIT_UPDATE=1

show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo "OPTIONS:"
    echo "  --work-dir DIR                  Set the work directory"
    echo "  --log-dir DIR                   Set the log directory"
    echo "  --concurrency NUM               Set the concurrency"
    echo "  --cpu NUM                       Set the CPU"
    echo "  --memory NUM                    Set the memory"
    echo "  --memory-profile                Enable memory profile"
    echo "  --make-bin \"[MAKE OPTIONS]\"     make-bin.sh arguments"
    echo "  --git-no-update                 Do not update git"
    echo "  -- [RUN OPTIONS]                docker-run-random.sh arguments"
}

while [ $# -gt 0 ]; do
    case "$1" in
    --work-dir)
        WORKDIR="$2"
        shift
        ;;
    --log-dir)
        LOGDIR="$2"
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
    --make-bin)
        MAKE_BIN_ARGS="$2"
        shift
        ;;
    --git-no-update)
        GIT_UPDATE=0
        ;;
    --)
        shift
        break
        ;;
    --help)
        show_help
        exit 0
        ;;
    *)
        show_help
        exit 1
        ;;
    esac
    shift
done

echo "================== Random Test for Replication $(date) =================="

set -x

source "$HOME/.cargo/env"

CWD=$(dirname "$(realpath -s "$0")")
cd "$CWD" || exit 1

if [ "$GIT_UPDATE" -eq 1 ]; then
    git pull
    git merge origin/cloud-engine --signoff --no-edit
fi
./make-bin.sh "$MAKE_BIN_ARGS"

mkdir -p "$WORKDIR"
mkdir -p "$LOGDIR"
export CONCURRENCY
export CPU
export MEMORY

declare -a RUN_ARGS
RUN_ARGS=(
    "--test" "replication"
    "--path-with-suffix"
    "--log-path" "$LOGDIR/rep-logs"
    "--tmp-path" "$WORKDIR/rep-tmp"
    "--no-tiflash"
    "--no-txn-file"
    "--no-jepsen"
    "--no-tpc"
    "--async-commit-ratio" "0.0"
    "--ia-table-ratio" "0.0"
)

if [ "$MEMORY_PROFILE" -eq 1 ]; then
    RUN_ARGS+=(
        "--memory-profile"
        "--keep-tmp-on-error"
    )
fi

./docker-stop-random.sh --test replication
./docker-run-random.sh "${RUN_ARGS[@]}" "$@"
