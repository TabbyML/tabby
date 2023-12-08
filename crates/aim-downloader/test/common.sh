#!/bin/bash

err() { echo -e "\e[1;31m${@}\e[0m" >&2; exit 1; }
ok() { echo -e "\e[1;32mOK\e[0m"; }
highlight() { echo -en "\e[1;37m${@}\e[0m"; }
function wait_docker() {
  until [ "`docker inspect -f {{.State.Health.Status}} $1`"=="healthy" ]; do
    sleep 1;
  done;
}

export PATH=$PATH:$(realpath ../../target/debug/)
