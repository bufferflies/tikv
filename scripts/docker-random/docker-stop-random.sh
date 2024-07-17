#!/bin/bash

set -euo pipefail

TESTNAME="all"

show_help() {
	echo "Usage: $0 [OPTIONS]"
	echo "OPTIONS:"
	echo "  --help                 Display this message"
	echo "  --test <all/with_tidb> Set the name of test case to stop"
}

while [[ $# -gt 0 ]]; do
	case "$1" in
	--test)
		TESTNAME="$2"
		shift
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

stop() {
	docker stop "$1" && docker rm "$1"
}

CONTAINERS=$(docker ps -aq --filter name=\^random-"$TESTNAME"-)

for container in $CONTAINERS; do
	stop "$container" &
done

wait
