#!/bin/bash

set -euo pipefail

CONCURRENCY="${CONCURRENCY:-8}"

TESTNAME="all"
HELP=0

while [[ $# -gt 0 ]]; do
	case "$1" in
	--test)
		TESTNAME="$2"
		shift
		;;
	--help)
		HELP=1
		break
		;;
	*)
		HELP=1
		break
		;;
	esac
	shift
done

if [ "$HELP" -eq 1 ]; then
	echo "Usage: $0 [OPTIONS]"
	echo "OPTIONS:"
	echo "  --help                 Display this message"
	echo "  --test <all/with_tidb> Set the name of test case to stop"
	exit 0
fi


stop() {
  docker stop random-"$TESTNAME"-"$1" && docker rm random-"$TESTNAME"-"$1"
}

for ((i=0;i<"$CONCURRENCY";i++)); do
  stop "$i" &
done

wait
