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

LOAD_DATA_TASK_TIMEOUT_SEC=90

UNIQUE_WORKLOAD=0
COLUMNAR_WORKLOAD=0

RESTART_TSO_SVC=1
IA_TABLE_RATIO=0.5
ASYNC_COMMIT_RATIO=0.1
TXN_CHECK_BACKUP_TS=1
OSS_CHAOS_RATIO=0.2

UPGRADE_TEST_DURATION="60s"

ENABLE_KV_ENGINE_META_DIFF=1
ENABLE_VALUE_CACHE=0
ENABLE_TIFLASH_WRITE_NODE=0

TIDB_NEXT_GEN=0

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
    --no-restart-tso-svc)
        RESTART_TSO_SVC=0
        ;;
    --ia-table-ratio)
        IA_TABLE_RATIO="$2"
        shift
        ;;
    --async-commit-ratio)
        ASYNC_COMMIT_RATIO="$2"
        shift
        ;;
    --no-check-backup-ts)
        TXN_CHECK_BACKUP_TS=0
        ;;
    --oss-chaos-ratio)
        OSS_CHAOS_RATIO="$2"
        shift
        ;;
    --upgrade-test-duration)
        UPGRADE_TEST_DURATION="$2"
        shift
        ;;
    --disable-kv-engine-meta-diff)
        ENABLE_KV_ENGINE_META_DIFF=0
        ;;
    --enable-value-cache)
        ENABLE_VALUE_CACHE=1
        ;;
    --enable-tiflash-write-node)
        ENABLE_TIFLASH_WRITE_NODE=1
        ;;
    --tidb-next-gen)
        TIDB_NEXT_GEN=1
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

export RESTART_TSO_SVC
export IA_TABLE_RATIO
export ASYNC_COMMIT_RATIO
export TXN_CHECK_BACKUP_TS
export OSS_CHAOS_RATIO

export TEST_DUR_BEFORE_UPGRADE="$UPGRADE_TEST_DURATION"
export TEST_DUR_AFTER_UPGRADE="$UPGRADE_TEST_DURATION"
export TEST_DUR_AFTER_DOWNGRADE="$UPGRADE_TEST_DURATION"

export ENABLE_KV_ENGINE_META_DIFF
export ENABLE_VALUE_CACHE
export ENABLE_TIFLASH_WRITE_NODE

export TIDB_NEXT_GEN

mkdir -p "$LOG_PATH"/logs "$LOG_PATH"/error-logs
for i in $(seq -w 1 100000); do
    export TMPDIR="/random-tmp/$i"
    mkdir -p "$TMPDIR"

    if [ "$MEMORY_PROFILE" -eq 1 ]; then
        export MALLOC_CONF="prof_leak:true,prof:true,lg_prof_interval:30,prof_final:true,prof_prefix:$TMPDIR/jeprof.out"
    fi

    LOG="$LOG_PATH"/logs/random_"$TESTNAME"_"$i"_"$DOCKER_ID".log
    CLUSTER_LOGS="$LOG_PATH"/logs/random_"$TESTNAME"_"$i"_"$DOCKER_ID"_tc
    mkdir -p "$CLUSTER_LOGS"
    # Random test get logs file path from env variable "LOG_FILE"
    export LOG_FILE="$CLUSTER_LOGS"/tikv.log
    export TEST_ID="$i"
    /random/random-bin test_random_"$TESTNAME" --nocapture >"$LOG" 2>&1 || true

    if [[ "$TESTNAME" =~ ^with_tidb|upgrade|replication$ ]]; then
        pkill -9 -f "/tidb-server" || true
        pkill -9 -f "/pd-server" || true
        pkill -9 -f "/tikv-server" || true
        pkill -9 -f "/tikv-worker" || true
        pkill -9 -f "/tiflash/tiflash" || true
        pkill -9 -f "/go-tpc" || true
        pkill -9 -f "minio" || true
        pkill -9 -f "/cdc" || true
    fi

    sync -d "$LOG"
    if grep -q 'TEST SUCCEED' "$LOG"; then
        grep 'TEST SUCCEED' "$LOG"
        rm "$LOG"
        rm -rf "$CLUSTER_LOGS" || true
        rm -rf "$TMPDIR" || true
    else
        sync -d "$LOG_FILE"
        grep -E 'CRIT|FATAL|panicked' "$LOG_FILE" >>"$LOG" || true

        # For logs of TiDB cluster components
        if compgen -G "$TMPDIR/tc*" >/dev/null; then
            chmod +r "$TMPDIR"/tc*/*.log
            cp "$TMPDIR"/tc*/*.log "$TMPDIR"/tc*/*.toml "$CLUSTER_LOGS" || true
        fi
        if compgen -G "$TMPDIR/rep*" >/dev/null; then
            chmod +r "$TMPDIR"/rep*/*.log
            cp "$TMPDIR"/rep*/*.log "$TMPDIR"/rep*/*.toml "$CLUSTER_LOGS" || true
        fi

        mv "$LOG" "$LOG_PATH"/error-logs/
        if [ "$KEEP_TMP_ON_ERROR" -ne 1 ]; then
            rm -rf "$TMPDIR" || true
        fi
    fi
done
