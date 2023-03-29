import os

import duckdb
import streamlit as st

st.set_page_config(page_title="Tabby Admin - Metrics")


def query_data():
    filepath = os.environ.get("DB_FILE", "/data/logs/duckdb/duck.db")
    if not os.path.isfile(filepath):
        return []

    conn = duckdb.connect(filepath)

    df = conn.sql(
        """
    SELECT
        date_trunc('hour', to_timestamp(CAST(created AS int64))) AS Date,
        SUM(IF(view, 1, 0)) as "Views",
        SUM(IF("select", 1, 0)) as "Acceptances"
    FROM completion_events
    GROUP BY 1;
    """
    ).df()

    conn.close()
    return df


df = query_data()

if len(df) > 0:
    st.markdown("### Completion Events")
    st.line_chart(df, x="Date")

    st.markdown("### Acceptance Rate")
    df["Acceptance Rate"] = df["Acceptances"] / df["Views"]
    st.line_chart(df, x="Date", y="Acceptance Rate")
else:
    st.markdown("No data available")
