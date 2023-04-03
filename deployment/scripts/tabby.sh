#!/bin/bash
set -e

# Shared environment variables
export LOGS_DIR="${LOGS_DIR:-/data/logs}"
export DB_FILE="${DB_FILE:-/data/logs/duckdb/duck.db}"
export CONFIG_FILE=${CONFIG_FILE:-/data/config/tabby.toml}

# server
export MODEL_NAME="${MODEL_NAME:-TabbyML/J-350M}"
export MODEL_BACKEND="${MODEL_BACKEND:-python}"

# dagu
export DAGU_DAGS="tabby/tasks"

init() {
if [ ! -f $CONFIG_FILE ]; then
  mkdir -p $(dirname $CONFIG_FILE)
  touch $CONFIG_FILE
fi

python -m tabby.tools.download_models --repo_id=$MODEL_NAME
}


supervisor() {
supervisord -n -c <(cat <<EOF
[supervisord]
logfile = /var/log/supervisord.log
loglevel = debug

[program:server]
command=uvicorn tabby.server:app --host 0.0.0.0 --port 5000

[program:admin]
command=streamlit run tabby/admin/Home.py --server.port 8501

[program:vector]
command=vector

[program:dagu_scheduler]
command=dagu scheduler

[program:dagu_server]
command=dagu server --host 0.0.0.0 --port 8080
EOF
)
}

init
supervisor
