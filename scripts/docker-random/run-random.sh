#!/bin/bash

set -eu

DOCKER_ID=$1
TESTNAME=$2
shift 2

# Optional keep temporary data on error for debugging.
# Note: When the container stops, the files from the last loop will not be automatically removed. However, they will be cleared during the next iterations.
KEEP_TMP_ON_ERROR=0
LOG_PATH="/random"
MEMORY_PROFILE=0

USE_TIFLASH=1
GLOBAL_TXN_FILE=1
USE_REMOTE_COP=1

TPC_WORKLOAD=1
TPCC_TXNS_THRESHOLD=100

JEPSEN_WORKLOAD=1
JEPSEN_TXN_FILE=1
JEPSEN_TXNS_THRESHOLD=100

LOAD_DATA_TASK_TIMEOUT_SEC=60

UNIQUE_WORKLOAD=0
COLUMNAR_WORKLOAD=0

while [ $# -gt 0 ]; do
    case "$1" in
    --keep-tmp-on-error)
        KEEP_TMP_ON_ERROR=1
        ;;
    --log-path)
        LOG_PATH="$2"
        shift
        ;;
    --memory-profile)
        MEMORY_PROFILE=1
        ;;
    --no-tiflash)
        USE_TIFLASH=0
        ;;
    --no-txn-file)
        GLOBAL_TXN_FILE=0
        ;;
    --no-remote-cop)
        USE_REMOTE_COP=0
        ;;
    --no-tpc)
        TPC_WORKLOAD=0
        ;;
    --tpcc-txns-threshold)
        TPCC_TXNS_THRESHOLD="$2"
        shift
        ;;
    --no-jepsen)
        JEPSEN_WORKLOAD=0
        ;;
    --jepsen-no-txn-file)
        JEPSEN_TXN_FILE=0
        ;;
    --jepsen-txns-threshold)
        JEPSEN_TXNS_THRESHOLD="$2"
        shift
        ;;
    --load-data-task-timeout-sec)
        LOAD_DATA_TASK_TIMEOUT_SEC="$2"
        shift
        ;;
    --unique-workload)
        UNIQUE_WORKLOAD=1
        ;;
    --columnar-workload)
        COLUMNAR_WORKLOAD=1
        ;;
    *)
        echo "Usage: $0 DOCKER_ID TESTNAME [--keep-tmp-on-error] [--log-path LOG_PATH] [--memory-profile]"
        exit 1
        ;;
    esac
    shift
done

export RUST_BACKTRACE=1
export LOG_LEVEL=info
# Components pattern for env_logger. E.g. export RUST_LOG="info,raft=debug"
export RUST_LOG="info"

export MEMORY_PROFILE

export USE_TIFLASH
export GLOBAL_TXN_FILE
export USE_REMOTE_COP

export TPC_WORKLOAD
export TPCC_TXNS_THRESHOLD

export JEPSEN_WORKLOAD
export JEPSEN_TXN_FILE
export JEPSEN_TXNS_THRESHOLD

export LOAD_DATA_TASK_TIMEOUT_SEC

export UNIQUE_WORKLOAD
export COLUMNAR_WORKLOAD

mkdir -p "$LOG_PATH"/logs "$LOG_PATH"/error-logs
for i in $(seq -w 1 100000); do
    export TMPDIR="/random-tmp/$i"
    mkdir -p "$TMPDIR"

    if [ "$MEMORY_PROFILE" -eq 1 ]; then
        export MALLOC_CONF="prof_leak:true,prof:true,lg_prof_interval:30,prof_final:true,prof_prefix:$TMPDIR/jeprof.out"
    fi

    LOG="$LOG_PATH"/logs/random_"$TESTNAME"_"$i"_"$DOCKER_ID".log
    /random/random-bin test_random_"$TESTNAME" >"$LOG" 2>&1 || true
    if grep -q 'TEST SUCCEED' "$LOG"; then
        grep 'TEST SUCCEED' "$LOG"
        rm "$LOG"
        rm -rf "$TMPDIR" || true
    else
        mv "$LOG" "$LOG_PATH"/error-logs/
        if [ "$KEEP_TMP_ON_ERROR" -ne 1 ]; then
            rm -rf "$TMPDIR" || true
        fi
    fi

    if [ "$TESTNAME" = "with_tidb" ] || [ "$TESTNAME" = "upgrade" ]; then
        pkill -9 tidb-server || true
        pkill -9 pd-server || true
        pkill -9 tikv-server || true
        pkill -9 tikv-worker || true
        pkill -9 tiflash || true
        pkill -9 go-tpc || true
    fi
done
