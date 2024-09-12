#!/bin/bash

HELP="Usage: $me [-b|--build]"

while [[ "$1" != "" ]]; do
    case "$1" in
    -h | --help)
        echo $HELP
        exit 0
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
    cp ~/.bashrc ./.bashrc
    docker compose build
    rm ./.bashrc
fi

docker compose run qtensor_dev
