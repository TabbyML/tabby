#!/bin/bash
set -e

DB_FILE=${DB_FILE:-"/data/logs/duckdb/duck.db"}
LOGS_DIR=${LOGS_DIR:-"/data/logs"}
TABBY_SERVER_LOGS="${LOGS_DIR}/tabby-server/events.*.json"

# Init schema
function init_scheme() {
mkdir -p $(dirname $DB_FILE)
cat <<EOF | duckdb
CREATE TABLE IF NOT EXISTS completion_events (
  id STRING,
  created uint64,
  prompt STRING,
  choices STRUCT(index UINT64, text STRING)[],
  view BOOLEAN,
  "select" BOOLEAN
);
CREATE UNIQUE INDEX IF NOT EXISTS completion_events_id ON completion_events (id);
EOF
}

# Update table
function collect_tabby_server_logs() {
cat <<EOF | duckdb
CREATE TEMP TABLE t AS
SELECT id, created, prompt, choices, IFNULL(rhs.view, false) AS view, IFNULL(rhs.select, false) AS select
FROM
  (
    SELECT
      id,
      FIRST(created) AS created,
      FIRST(prompt) AS prompt,
      FIRST(choices) AS choices
    FROM '${TABBY_SERVER_LOGS}' WHERE id IS NOT NULL GROUP BY 1) lhs
LEFT JOIN (
    SELECT
      completion_id,
      (SUM(IF(type == 'view', 1, 0)) > 0) AS view,
      (SUM(IF(type == 'select', 1, 0)) > 0) AS select
    FROM '${TABBY_SERVER_LOGS}'
    WHERE completion_id IS NOT NULL
    GROUP BY 1
) rhs ON (lhs.id = rhs.completion_id);

INSERT INTO completion_events SELECT t.* FROM t LEFT JOIN completion_events rhs ON (t.id = rhs.id) WHERE rhs.id IS NULL;
EOF
}

function duckdb() {
  local SQL=$(tee)
  cat << EOF | python3 -
import sys
import duckdb
conn = duckdb.connect('$DB_FILE')
print(conn.sql("""
$SQL
"""))
EOF
}

init_scheme
"$@"
