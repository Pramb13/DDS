import streamlit as st
import requests
import pandas as pd
import random

st.set_page_config(page_title="DriveGuardian", layout="wide")
st.sidebar.title("🔐 Login")

user = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type="password")
login_btn = st.sidebar.button("Login")

USERS = {
    "admin": {"password": "1234", "role": "admin"},
    "driver1": {"password": "abcd", "role": "user"},
    "driver2": {"password": "qwerty", "role": "user"},
}

if login_btn:
    if user in USERS and password == USERS[user]["password"]:
        role = USERS[user]["role"]
        st.sidebar.success(f"Logged in as {role.capitalize()}")
        st.title(f"🚗 DriveGuardian - {role.capitalize()} Dashboard")

        if role == "admin":
            st.subheader("📊 Driver Behavior Analytics")
            data = {
                "Driver": ["driver1", "driver2", "driver1", "driver2"],
                "Date": pd.date_range(end=pd.Timestamp.today(), periods=4).date,
                "Status": random.choices(["Alert", "Drowsy", "Distracted"], k=4),
                "Yawns": [random.randint(1, 10) for _ in range(4)],
                "Session Hours": [round(random.uniform(1, 4), 2) for _ in range(4)],
            }
            df = pd.DataFrame(data)
            st.dataframe(df)
            st.bar_chart(df.groupby("Driver")["Yawns"].sum())
            st.line_chart(df.pivot(index="Date", columns="Driver", values="Session Hours"))

        elif role == "user":
            st.subheader("👤 Driver Assistant")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Check My Drowsiness"):
                    res = requests.get("http://localhost:8000/predict_drowsiness")
                    st.metric("Your Current Status", res.json()["status"])
            with col2:
                st.info("Ask your Assistant (RAG)")
                question = st.text_input("Ask something about your driving history")
                if question:
                    response = requests.post("http://localhost:8000/query", json={"question": question})
                    st.success(response.json()["answer"])
    else:
        st.sidebar.error("Invalid username or password.")
