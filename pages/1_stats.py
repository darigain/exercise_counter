# pages/Stats.py

import streamlit as st
import psycopg2
import pandas as pd
import plotly.express as px
import datetime

st.title("Exercise Statistics")

@st.cache_data(show_spinner=False)
def load_data():
    """
    Connect to the database and load exercise_records data into a DataFrame.
    """
    try:
        conn = psycopg2.connect(
            host=st.secrets["db"]["host"],
            database=st.secrets["db"]["database"],
            user=st.secrets["db"]["user"],
            password=st.secrets["db"]["password"],
            port=st.secrets["db"]["port"]
        )
        query = "SELECT username, datetime, squat_count, pushup_count FROM exercise_records ORDER BY datetime"
        df = pd.read_sql_query(query, conn)
        conn.close()
        # Ensure that the datetime column is in datetime format.
        df['datetime'] = pd.to_datetime(df['datetime'])
        return df
    except Exception as e:
        st.error(f"Error retrieving data: {e}")
        return pd.DataFrame()

# Load data from the database.
df = load_data()

if df.empty:
    st.warning("No data found in the database.")
    st.stop()

# Sidebar widgets for filtering

# 1. Drop-down list to filter by user.
user_options = sorted(df['username'].unique().tolist())
selected_user = st.selectbox("Select a user", ["All Users"] + user_options)

if selected_user != "All Users":
    df = df[df['username'] == selected_user]

# 2. Drop-down list for aggregation frequency.
frequency = st.selectbox("Select aggregation frequency", ["Daily", "Weekly", "Monthly"])

# Prepare the data for plotting based on selected frequency.
if frequency == "Daily":
    # Create a 'date' column from the datetime column.
    df['date'] = df['datetime'].dt.date
    df_grouped = (
        df.groupby("date")[["squat_count", "pushup_count"]]
          .sum()
          .reset_index()
    )
    # Convert date back to datetime for consistent plotting.
    df_grouped['date'] = pd.to_datetime(df_grouped['date'])
    
elif frequency == "Weekly":
    # Create a 'week' column corresponding to the start date of the week.
    df['week'] = df['datetime'].dt.to_period('W').apply(lambda r: r.start_time)
    df_grouped = (
        df.groupby("week")[["squat_count", "pushup_count"]]
          .sum()
          .reset_index()
          .rename(columns={'week': 'date'})
    )
    
elif frequency == "Monthly":
    # Create a 'month' column corresponding to the start date of the month.
    df['month'] = df['datetime'].dt.to_period('M').apply(lambda r: r.start_time)
    df_grouped = (
        df.groupby("month")[["squat_count", "pushup_count"]]
          .sum()
          .reset_index()
          .rename(columns={'month': 'date'})
    )
else:
    df_grouped = df.copy()

# Create an interactive line plot with two lines (one for squat_count, one for pushup_count).
fig = px.line(
    df_grouped,
    x="date",
    y=["squat_count", "pushup_count"],
    markers=True,
    labels={"value": "Count", "date": "Date", "variable": "Exercise"},
    title=f"Exercise Counts ({frequency} Aggregation)"
)

st.plotly_chart(fig, use_container_width=True)
