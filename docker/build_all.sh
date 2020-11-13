#!/bin/bash
#
# Build latest:
# ./build_all.sh
#
# Build and push version X.Y.Z:
# ./build_all.sh X.Y.Z 1

set -e

# allow user to run this script from anywhere
# from https://stackoverflow.com/a/246128
# one-liner which will give you the full directory name
# of the script no matter where it is being called from
unset CDPATH
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

ROOT_DIR=$DIR/..
cd $ROOT_DIR

VERSION=${1:-latest}
PUSH=${2:-0}
IMAGE=opennmt/ctranslate2

build()
{
    DOCKERFILE=$1
    IMAGE_NAME=$2
    BUILD_ARGS=${3:-}

    LATEST=$IMAGE:latest-$IMAGE_NAME
    TAGGED=$IMAGE:$VERSION-$IMAGE_NAME
    docker build --pull $BUILD_ARGS -t $LATEST -f docker/$DOCKERFILE .
    if [ $PUSH -eq 1 ]; then
        docker push $LATEST
    fi
    if [ "$TAGGED" != "$LATEST" ]; then
        docker tag $LATEST $TAGGED
        if [ $PUSH -eq 1 ]; then
            docker push $TAGGED
        fi
    fi
}

build Dockerfile.centos7 centos7
build Dockerfile.centos7-gpu centos7-gpu "--build-arg CUDA_VERSION=10.0"
build Dockerfile.centos7-gpu centos7-cuda10.0 "--build-arg CUDA_VERSION=10.0"
build Dockerfile.centos7-gpu centos7-cuda10.1 "--build-arg CUDA_VERSION=10.1"
build Dockerfile.centos7-gpu centos7-cuda10.2 "--build-arg CUDA_VERSION=10.2"
build Dockerfile.centos7-gpu centos7-cuda11.0 "--build-arg CUDA_VERSION=11.0"

build Dockerfile.ubuntu ubuntu18
build Dockerfile.ubuntu-gpu ubuntu18-gpu "--build-arg CUDA_VERSION=10.0"
build Dockerfile.ubuntu-gpu ubuntu18-cuda10.0 "--build-arg CUDA_VERSION=10.0"
build Dockerfile.ubuntu-gpu ubuntu18-cuda10.1 "--build-arg CUDA_VERSION=10.1"
build Dockerfile.ubuntu-gpu ubuntu18-cuda10.2 "--build-arg CUDA_VERSION=10.2"
build Dockerfile.ubuntu-gpu ubuntu18-cuda11.0 "--build-arg CUDA_VERSION=11.0"
