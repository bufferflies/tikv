#!/bin/bash

# Usage:
#  CONCURRENCY=8 ./docker-run-random.sh

set -euo pipefail

CONCURRENCY="${CONCURRENCY:-8}"
# Tend to use less CPU to generate more process context switches.
CPU="${CPU:-4}"
MEMORY="${MEMORY:-5g}"

TMP_PATH=""
LOG_PATH=""
PATH_SUFFIX=""
TESTNAME="all"
TIDB_VERSION="v8.5.3"
TIKV_VERSION=""
REBUILD_IMAGE=1
declare -a RUN_ARGS
RUN_ARGS=()
KEEP_TMP_ON_ERROR=0
MEMORY_PROFILE=0

show_help() {
	echo "Usage: $0 [OPTIONS]"
	echo "OPTIONS:"
	echo "  --help                                 Display this message"
	echo "  --tmp-path     <temporary path>        Set the path for temporary data generated during testing"
	echo "  --log-path     <log path>              Set the path for logs"
	echo "  --path-with-suffix                     Add git commit and timestamp to the path as suffix"
	echo "  --test         <all/with_tidb/upgrade> Set the name of test case to run"
	echo "  --tidb-version <v6.6.0/v7.1.0/...>     Set the version of TiDB for \"with_tidb\" test"
	echo "  --tikv-version <git_hash>              Set the version of old TiKV for \"upgrade\" test"
	echo "  --no-rebuild-image                     Do NOT rebuild the testing Docker image"
	echo "  --keep-tmp-on-error                    Keep temporary data on error for debugging"
	echo "  --memory-profile                       Enable memory profiling"
}

PWD=$(pwd)

while [[ $# -gt 0 ]]; do
	case "$1" in
	--tmp-path)
		TMP_PATH="$2"
		shift
		;;
	--log-path)
		LOG_PATH="$2"
		shift
		;;
	--path-with-suffix)
		PATH_SUFFIX=-$(date +%Y%m%d-%H%M%S)-$(git rev-parse --short HEAD)
		;;
	--test)
		TESTNAME="$2"
		shift
		;;
	--tidb-version)
		TIDB_VERSION="$2"
		shift
		;;
	--tikv-version)
		TIKV_VERSION="$2"
		shift
		;;
	--no-rebuild-image)
		REBUILD_IMAGE=0
		;;
	--keep-tmp-on-error)
		KEEP_TMP_ON_ERROR=1
		RUN_ARGS+=("--keep-tmp-on-error")
		;;
	--memory-profile)
		MEMORY_PROFILE=1
		RUN_ARGS+=("--memory-profile")
		;;
	--no-tiflash)
		RUN_ARGS+=("--no-tiflash")
		;;
	--no-txn-file)
		RUN_ARGS+=("--no-txn-file")
		;;
	--no-remote-cop)
		RUN_ARGS+=("--no-remote-cop")
		;;
	--no-tpc)
		RUN_ARGS+=("--no-tpc")
		;;
	--tpcc-txns-threshold)
		RUN_ARGS+=("--tpcc-txns-threshold" "$2")
		shift
		;;
	--no-jepsen)
		RUN_ARGS+=("--no-jepsen")
		;;
	--jepsen-no-txn-file)
		RUN_ARGS+=("--jepsen-no-txn-file")
		;;
	--jepsen-txns-threshold)
		RUN_ARGS+=("--jepsen-txns-threshold" "$2")
		shift
		;;
	--load-data-task-timeout-sec)
		RUN_ARGS+=("--load-data-task-timeout-sec" "$2")
		shift
		;;
	--unique-workload)
		RUN_ARGS+=("--unique-workload")
		;;
	--columnar-workload)
		RUN_ARGS+=("--columnar-workload")
		;;
	--no-restart-tso-svc)
		RUN_ARGS+=("--no-restart-tso-svc")
		;;
	--ia-table-ratio)
		RUN_ARGS+=("--ia-table-ratio" "$2")
		shift
		;;
	--async-commit-ratio)
		RUN_ARGS+=("--async-commit-ratio" "$2")
		shift
		;;
	--no-check-backup-ts)
		RUN_ARGS+=("--no-check-backup-ts")
		;;
	--oss-chaos-ratio)
		RUN_ARGS+=("--oss-chaos-ratio" "$2")
		shift
		;;
	--upgrade-test-duration)
		RUN_ARGS+=("--upgrade-test-duration" "$2")
		shift
		;;
	--disable-value-cache)
		RUN_ARGS+=("--disable-value-cache")
		;;
	--enable-tiflash-write-node)
		RUN_ARGS+=("--enable-tiflash-write-node")
		;;
	--disable-columnar-normal-workload)
		RUN_ARGS+=("--disable-columnar-normal-workload")
		;;
	--disable-columnar-dynamic-workload)
		RUN_ARGS+=("--disable-columnar-dynamic-workload")
		;;
	--disable-columnar-partition-workload)
		RUN_ARGS+=("--disable-columnar-partition-workload")
		;;
	--tidb-next-gen)
		RUN_ARGS+=("--tidb-next-gen")
		;;
	--rep-debug-log)
		RUN_ARGS+=("--rep-debug-log")
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

if [ "$MEMORY_PROFILE" -eq 1 ] && [ "$KEEP_TMP_ON_ERROR" -ne 1 ]; then
	echo "WARNING: --keep-tmp-on-error is not enabled. The profile dumps will be removed after each test."
fi

BUILD_IMAGE_ARGS=""
if [ "$REBUILD_IMAGE" -eq 1 ]; then
	BUILD_IMAGE_ARGS+=" --pull --no-cache"
fi

IMAGE="amazonlinux:2023.5.20241001.1"
if [[ "$TESTNAME" =~ ^with_tidb|replication$ ]]; then
	docker build $BUILD_IMAGE_ARGS -t random-tidb --build-arg TIDB_VERSION="$TIDB_VERSION" - <Dockerfile.tidb
	IMAGE="random-tidb"
elif [ "$TESTNAME" = "upgrade" ]; then
	if [ -z "$TIKV_VERSION" ]; then
		echo "ERROR: --tikv-version is required for upgrade test"
		exit 1
	fi
	docker build $BUILD_IMAGE_ARGS -t random-upgrade \
		--build-arg TIDB_VERSION="$TIDB_VERSION" \
		--build-arg TIKV_VERSION="$TIKV_VERSION" \
		- <Dockerfile.upgrade
	IMAGE="random-upgrade"
elif [ "$MEMORY_PROFILE" -eq 1 ]; then
	# Parse profile dumps in an environment different with container for random test would fail to translate the addresses to symbols.
	# So provide an image with tools.
	docker build -t random-profile - <Dockerfile.profile
	IMAGE="random-profile"
fi

LOG_VOLUME=""
if [ -n "$LOG_PATH" ]; then
	ABSOLUTE_LOG_PATH=$(readlink -f "${LOG_PATH}${PATH_SUFFIX}")
	LOG_VOLUME="-v ${ABSOLUTE_LOG_PATH}:/random-logs"
	RUN_ARGS+=("--log-path" "/random-logs")
fi

for ((i = 0; i < "$CONCURRENCY"; i++)); do
	TMP_VOLUME=""
	if [ -n "$TMP_PATH" ]; then
		# Containers should use different $TMPDIR, otherwise they will conflict on TIKV_LOCK_FILES.
		mkdir -p "${TMP_PATH}${PATH_SUFFIX}/${i}"
		ABSOLUTE_TMP_PATH=$(readlink -f "${TMP_PATH}${PATH_SUFFIX}/$i")
		TMP_VOLUME="-v $ABSOLUTE_TMP_PATH:/random-tmp"
	fi

	docker run --name random-"$TESTNAME"-"$i" --cpus="$CPU" --memory="$MEMORY" \
		-itd \
		-v "$PWD":/random \
		$TMP_VOLUME \
		$LOG_VOLUME \
		"$IMAGE" \
		/bin/bash /random/run-random.sh "$i" "$TESTNAME" "${RUN_ARGS[@]}"
done
