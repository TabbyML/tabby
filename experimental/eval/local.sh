#!/bin/bash

TABBY_ROOT=$PWD/tabby cargo run scheduler --now

./eval.sh
