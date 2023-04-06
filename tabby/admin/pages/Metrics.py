import os

import duckdb
import streamlit as st
from utils.streamlit import set_page_config

set_page_config(page_title="Metrics")


def query_data():
    filepath = os.environ.get("DB_FILE", "/data/logs/duckdb/duck.db")
    if not os.path.isfile(filepath):
        return []

    conn = duckdb.connect(filepath)

    df = conn.sql(
        """
    SELECT
        date_trunc('day', to_timestamp(CAST(created AS int64))) AS Date,
        SUM(IF(view, 1, 0)) as "Views",
        SUM(IF("select", 1, 0)) as "Acceptances"
    FROM completion_events
    GROUP BY 1;
    """
    ).df()

    conn.close()
    return df


df = query_data()


def plot_summary():
    sum_views = int(sum(df.Views))
    sum_acceptances = int(sum(df.Acceptances))
    ratio = (sum_acceptances / sum_views) * 100

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Views", sum_views)
    with col2:
        st.metric("Acceptances", sum_acceptances)
    with col3:
        st.metric("Accept Rate", f"{round(ratio)} %")


def plot_charts():
    st.markdown("### Completion Events")
    st.line_chart(df, x="Date")

    st.markdown("### Accept Rate")
    df["Acceptance Rate"] = df["Acceptances"] / df["Views"]
    st.line_chart(df, x="Date", y="Acceptance Rate")


if len(df) > 0:
    plot_summary()
    st.write("---")
    plot_charts()
else:
    st.markdown("No data available")
