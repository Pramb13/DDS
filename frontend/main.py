import streamlit as st
import requests
import pandas as pd
import random

# Set page config and title
st.set_page_config(page_title="DRIVER DROWSINESS DETECTION", layout="wide")
st.markdown("<h1 style='text-align: center; color: #2E86C1;'>🚘 DRIVER DROWSINESS DETECTION</h1>", unsafe_allow_html=True)

# Sidebar Login
st.sidebar.title("🔐 Login")
user = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type="password")
login_btn = st.sidebar.button("Login")

# Dummy users
USERS = {
    "admin": {"password": "1234", "role": "admin"},
    "driver1": {"password": "abcd", "role": "user"},
    "driver2": {"password": "xyz", "role": "user"},
}

if login_btn:
    if user in USERS and password == USERS[user]["password"]:
        role = USERS[user]["role"]
        st.sidebar.success(f"Logged in as {role.capitalize()}")
        st.title(f"🚗 DRIVER DROWSINESS DETECTION - {role.capitalize()} Dashboard")

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
            st.subheader("👁️ Real-Time Drowsiness Detection")

            try:
                res = requests.get("http://localhost:8000/predict_drowsiness")
                status = res.json()["status"]
                st.success(f"🚨 Your Current Status: {status}")
            except Exception as e:
                st.error(f"Detection failed: {e}")

            # Floating chat button and chat box
            st.markdown("""
                <style>
                    #chat-btn {
                        position: fixed;
                        bottom: 30px;
                        right: 30px;
                        background-color: #2980B9;
                        color: white;
                        padding: 12px 20px;
                        border: none;
                        border-radius: 50px;
                        cursor: pointer;
                        font-size: 16px;
                        z-index: 9999;
                    }
                    #chat-box {
                        position: fixed;
                        bottom: 80px;
                        right: 30px;
                        width: 300px;
                        background-color: white;
                        border: 1px solid #ccc;
                        border-radius: 10px;
                        padding: 10px;
                        display: none;
                        z-index: 9999;
                    }
                </style>
                <script>
                    function toggleChat() {
                        var chat = document.getElementById("chat-box");
                        if (chat.style.display === "none") {
                            chat.style.display = "block";
                        } else {
                            chat.style.display = "none";
                        }
                    }
                </script>
                <button id="chat-btn" onclick="toggleChat()">💬 Chat Assistant</button>
                <div id="chat-box">
                    <form action="" method="post">
                        <input type="text" name="question" id="rag-input" placeholder="Ask me anything..." style="width: 100%; padding: 5px;" />
                    </form>
                </div>
            """, unsafe_allow_html=True)

            # RAG input field inside chat box (still shown in main app)
            question = st.text_input("💬 Assistant (RAG)", key="rag-input-box")
            if question:
                try:
                    response = requests.post("http://localhost:8000/query", json={"question": question})
                    st.success(response.json()["answer"])
                except Exception as e:
                    st.error(f"Assistant error: {e}")
    else:
        st.sidebar.error("Invalid username or password.")
