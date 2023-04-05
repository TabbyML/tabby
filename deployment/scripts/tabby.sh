#!/bin/bash
set -e

# Shared environment variables
export LOGS_DIR="${LOGS_DIR:-/data/logs}"
export DB_FILE="${DB_FILE:-/data/logs/duckdb/duck.db}"
export CONFIG_FILE=${CONFIG_FILE:-/data/config/tabby.toml}

# server
export MODEL_NAME="${MODEL_NAME:-TabbyML/J-350M}"
export MODEL_BACKEND="${MODEL_BACKEND:-python}"

# projects
export GIT_REPOSITORIES_DIR="${REPOSITORIES_DIR:-/data/repositories}"
export DATASET_DIR="${REPOSITORIES_DIR:-/data/dataset}"

# dagu
export DAGU_DAGS="tabby/tasks"

init() {
if [ ! -f $CONFIG_FILE ]; then
  mkdir -p $(dirname $CONFIG_FILE)
  touch $CONFIG_FILE
fi

# Disable safe directory check
git config --global --add safe.directory '*'

python -m tabby.tools.download_models --repo_id=$MODEL_NAME
}


supervisor() {
if [[ "$MODEL_BACKEND" == "triton" ]]
then

local TRITON_SERVER=$(cat <<EOF
[program:triton]
command=triton.sh
EOF
)

fi

supervisord -n -c <(cat <<EOF
[supervisord]
logfile = ${LOGS_DIR}/supervisord.log
loglevel = debug

[program:server]
command=uvicorn tabby.server:app --host 0.0.0.0 --port 5000

[program:admin]
command=streamlit run tabby/admin/Home.py --server.port 8501 --theme.base=dark

[program:vector]
command=vector --config-toml tabby/config/vector.toml

[program:dagu_scheduler]
command=dagu scheduler

[program:dagu_server]
command=dagu server --host 0.0.0.0 --port 8080

$TRITON_SERVER
EOF
)
}

init
supervisor
