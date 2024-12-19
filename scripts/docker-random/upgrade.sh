#!/bin/bash

set -euo pipefail
CONCURRENCY=4
CPU=4
MEMORY=8g
MAKE_BIN_ARGS=""
GIT_UPDATE=1
UPGRADE_FROM=""

WORKDIR="/data/nvme1n1/$LOGNAME/random"

show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo "OPTIONS:"
    echo "  --work-dir DIR                  Set the work directory"
    echo "  --concurrency NUM               Set the concurrency"
    echo "  --cpu NUM                       Set the CPU"
    echo "  --memory NUM                    Set the memory"
    echo "  --make-bin [MAKE OPTIONS]       make-bin.sh arguments"
    echo "  --upgrade-from                  Upgrade from the specific version"
    echo "  --git-no-update                 Do not update git"
    echo "  -- [RUN OPTIONS]                docker-run-random.sh arguments"
}

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
    --upgrade-from)
        UPGRADE_FROM="$2"
        shift
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

echo "================== Random Test for Upgrade $(date) =================="

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

export CONCURRENCY
export CPU
export MEMORY

./docker-stop-random.sh --test upgrade
./docker-run-random.sh \
    --path-with-suffix \
    --keep-tmp-on-error \
    --test upgrade \
    --tikv-version "$UPGRADE_FROM" \
    --log-path "$WORKDIR/upgrade-logs" \
    --tmp-path "$WORKDIR/upgrade-tmp" \
    "$@"
