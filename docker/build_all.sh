#!/bin/sh

VERSION=${1:-latest}
PUSH=${2:-0}

build()
{
    PLAT=$1
    IMAGE=systran/ctranslate2:$VERSION-$PLAT
    docker build -t $IMAGE -f docker/Dockerfile.$PLAT .
    if [ $PUSH -eq 1 ]; then
        docker push $IMAGE
    fi
}

build ubuntu16
build centos7
build centos7-gpu
