import streamlit as st
import requests
import pandas as pd
import random

# Set page config and top title
st.set_page_config(page_title="DRIVER DROWSINESS DETECTION", layout="wide")
st.markdown(
    "<h1 style='text-align: center; color: #2E86C1;'>🚘 DRIVER DROWSINESS DETECTION</h1>",
    unsafe_allow_html=True
)

# Sidebar Login
st.sidebar.title("🔐 Login")
username = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type="password")
login_btn = st.sidebar.button("Login")

# Dummy user database
USERS = {
    "admin": {"password": "1234", "role": "admin"},
    "driver1": {"password": "abcd", "role": "user"},
    "driver2": {"password": "xyz", "role": "user"},
}

if login_btn:
    if username in USERS and password == USERS[username]["password"]:
        role = USERS[username]["role"]
        st.sidebar.success(f"✅ Logged in as {role.capitalize()}")

        st.subheader(f"🚗 Welcome, {username}!")
        st.markdown(f"### 🧭 {role.capitalize()} Dashboard")

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
                if st.button("🧪 Check My Drowsiness"):
                    try:
                        res = requests.get("http://localhost:8000/predict_drowsiness")
                        if res.status_code == 200:
                            st.metric("Your Current Status", res.json().get("status", "Unknown"))
                        else:
                            st.error("Failed to get status: Server returned an error.")
                    except Exception as e:
                        st.error(f"Connection error: {e}")

            with col2:
                st.info("💬 Ask your Assistant (RAG)")
                question = st.text_input("Ask something about your driving history:")
                if question:
                    try:
                        response = requests.post("http://localhost:8000/query", json={"question": question})
                        if response.status_code == 200:
                            st.success(response.json().get("answer", "No response"))
                        else:
                            st.error("Server error while getting response.")
                    except Exception as e:
                        st.error(f"Assistant connection failed: {e}")
    else:
        st.sidebar.error("❌ Invalid username or password.")
