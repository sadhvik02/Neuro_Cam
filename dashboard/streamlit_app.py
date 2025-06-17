import pandas as pd
import streamlit as st

st.title("🧠 NeuroCam Fatigue Dashboard")

# Load data
df = pd.read_csv("data/log.csv", names=["Timestamp", "Status"])

# Show last 20 entries
st.subheader("📋 Fatigue Events")
st.dataframe(df.tail(20))

# Add pie chart
st.subheader("📊 Fatigue Summary")
st.write(df["Status"].value_counts())
st.bar_chart(df["Status"].value_counts())
