#!/bin/bash

HELP="Usage: $me [-b|--build]"
SERVICE="qtensor_dev"
COMMAND=""

while [[ "$1" != "" ]]; do
    case "$1" in
    -h | --help)
        echo $HELP
        exit 0
        ;;
    -s | --service)
        SERVICE=$2
        shift 2
        ;;
    -c | --command)
        COMMAND=$2
        shift 2
        ;;
    -b | --build)
        BUILD=true
        shift
        ;;
    *)
        echo "Invalid argument: $1"
        echo $HELP
        exit 1
        ;;
    esac
done

# set env vars
export USER=$(whoami)
export USERID=$(id -u)
export GROUP=$(id -g -n)
export GROUPID=$(id -g)

set -o errexit
cd $(dirname "$(readlink -f "$0")")

if [ "$BUILD" = true ]; then
    docker compose build
fi

echo "Running $SERVICE with command: $COMMAND"

if [ -n "$COMMAND" ]; then
    docker compose run --rm $SERVICE /bin/bash -c "$COMMAND"
else
    docker compose run --rm $SERVICE
fi
