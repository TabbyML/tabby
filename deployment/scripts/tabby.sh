#!/bin/bash
set -e

# Shared environment variables
export LOGS_DIR="${LOGS_DIR:-/data/logs}"
export DB_FILE="${DB_FILE:-/data/logs/duckdb/duck.db}"
export CONFIG_FILE=${CONFIG_FILE:-/data/config/tabby.toml}

# server
export MODEL_NAME="${MODEL_NAME:-TabbyML/J-350M}"
export MODEL_BACKEND="${MODEL_BACKEND:-python}"
if [ ! -n "$WEB_CONCURRENCY" ]; then
  # Set WEB_CONCURRENCY for uvicorn workers.
  export WEB_CONCURRENCY=$(nproc --all)
fi

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
supervisord -n -c <(cat <<EOF
[supervisord]
logfile = /var/log/supervisord.log
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
EOF
)
}

init
supervisor
