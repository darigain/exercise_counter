# pages/Leaderboard.py

import streamlit as st
import psycopg2
import pandas as pd
import plotly.express as px
import datetime

st.title("Leaderboard")

st.write("🏆 Champions, you are absolutely crushing it! 🌟")

# Refresh button
if st.button("Refresh Data"):
    st.experimental_rerun()  # Force a rerun so that new data is loaded.

# ------------------------------
# TIMEFRAME FILTER SELECTION
# ------------------------------
# Options in exact order: Last 7 days, Last 30 days, All Time (default: Last 7 days)
timeframe = st.selectbox(
    "Select Timeframe",
    options=["Last 7 days", "Last 30 days", "All Time"],
    index=0
)

# ------------------------------
# DATA LOADING FUNCTION
# ------------------------------
@st.cache_data(show_spinner=False)
def load_data():
    """
    Connects to the database and loads data from the exercise_records table into a DataFrame.
    """
    try:
        conn = psycopg2.connect(
            host=st.secrets["db"]["host"],
            database=st.secrets["db"]["database"],
            user=st.secrets["db"]["user"],
            password=st.secrets["db"]["password"],
            port=st.secrets["db"]["port"]
        )
        query = "SELECT username, datetime, squat_count, pushup_count FROM exercise_records"
        df = pd.read_sql_query(query, conn)
        conn.close()
        df['datetime'] = pd.to_datetime(df['datetime'])
        return df
    except Exception as e:
        st.error(f"Error retrieving data: {e}")
        return pd.DataFrame()

# Load data from the database.
df = load_data()

if df.empty:
    st.warning("No data available from the database.")
    st.stop()

# ------------------------------
# APPLY TIMEFRAME FILTER
# ------------------------------
now = datetime.datetime.now()
if timeframe == "Last 7 days":
    threshold = now - datetime.timedelta(days=7)
    df = df[df['datetime'] >= threshold]
elif timeframe == "Last 30 days":
    threshold = now - datetime.timedelta(days=30)
    df = df[df['datetime'] >= threshold]
# For "All Time", no filtering is applied.

# ------------------------------
# AGGREGATE DATA BY USER
# ------------------------------
agg_df = df.groupby("username", as_index=False)[["squat_count", "pushup_count"]].sum()
agg_df["total_count"] = agg_df["squat_count"] + agg_df["pushup_count"]
agg_df = agg_df.sort_values(by="total_count", ascending=False)

# ------------------------------
# VISUALIZATION: TOP 10 USERS HORIZONTAL BAR CHART
# ------------------------------
top10 = agg_df.head(10)
# Melt the data for a stacked bar chart
top10_melt = top10.melt(
    id_vars="username",
    value_vars=["squat_count", "pushup_count"],
    var_name="Exercise",
    value_name="Count"
)
fig = px.bar(
    top10_melt,
    y="username",
    x="Count",
    color="Exercise",
    orientation="h",
    title="Top 10 Users Leaderboard (Stacked Counts)",
    labels={"username": "User", "Count": "Exercise Count"}
)
# Task 1: Reverse the y-axis order so that the highest total is at the top.
fig.update_yaxes(autorange="reversed")
st.plotly_chart(fig, use_container_width=True)

# ------------------------------
# TABLE: TOP 100 USERS LEADERBOARD
# ------------------------------
top100 = agg_df.head(100).copy()
top100 = top100.reset_index(drop=True)
top100["Rank"] = top100.index + 1
top100 = top100[["Rank", "username", "total_count", "squat_count", "pushup_count"]]

st.subheader("Top 100 Users Leaderboard")
# Task 2: Render the table as HTML without the default index column.
html_table = top100.to_html(index=False)
st.markdown(html_table, unsafe_allow_html=True)
