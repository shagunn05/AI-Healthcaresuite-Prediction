import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import streamlit as st
import os
from keras.models import load_model

st.set_page_config(page_title="RNN/LSTM/GRU Suite", page_icon="📊")
st.title("Sequential / Time-Series Prediction")

# Dropdown for Model Selection
model_choice = st.selectbox("Apna Model Select Karein:", ["RNN", "LSTM", "GRU"])

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# User ke choice ke hisab se sahi file select karein
if model_choice == "RNN":
    model_name = "rnn_healthcare_model.keras"
elif model_choice == "LSTM":
    model_name = "lstm_healthcare_model.keras"
else:
    model_name = "gru_healthcare_model.keras"

MODEL_PATH = os.path.join(BASE_DIR, "models", model_name)

# Selected Model load karein
try:
    selected_model = load_model(MODEL_PATH)
    st.success(f"{model_choice} Model Loaded Successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")

# --- Yahan apna Data Input aur Prediction Logic likhein ---

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Health Dashboard", layout="wide")

st.title("📊 RNN vs LSTM vs GRU - Health Prediction Dashboard")
st.write("Use sliders to input patient data and compare models.")


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RNN_MODEL_PATH = os.path.join(BASE_DIR, "models", "rnn_healthcare_model.keras")
LSTM_MODEL_PATH = os.path.join(BASE_DIR, "models", "lstm_healthcare_model.keras")
GRU_MODEL_PATH = os.path.join(BASE_DIR, "models", "gru_healthcare_model.keras")

rnn_model = load_model(RNN_MODEL_PATH)
lstm_model = load_model(LSTM_MODEL_PATH)
gru_model = load_model(GRU_MODEL_PATH)

# ---------------- INPUTS ----------------
st.subheader("📥 Patient Input Features")

col1, col2, col3 = st.columns(3)

with col1:
    glucose = st.slider("Glucose", 50, 300, 120)
    heart_rate = st.slider("Heart Rate", 40, 180, 80)
    stress_level = st.slider("Stress Level", 0, 10, 5)

with col2:
    exercise_steps = st.slider("Exercise Steps", 0, 20000, 5000)
    sleep_stage = st.slider("Sleep Stage (0-3)", 0, 3, 1)
    age = st.slider("Age", 10, 100, 30)

with col3:
    weight = st.slider("Weight (kg)", 30, 150, 65)
    glucose_lag_1 = st.slider("Glucose Lag 1", 50, 300, 110)
    glucose_roll_mean = st.slider("Glucose Rolling Mean", 50, 300, 115)

# ---------------- FEATURE VECTOR ----------------
features = [
    glucose,
    heart_rate,
    stress_level,
    exercise_steps,
    sleep_stage,
    age,
    weight,
    glucose_lag_1,
    glucose_roll_mean
]

# IMPORTANT: (1,10,9)
sequence = np.array([[features] * 10], dtype=np.float32)

st.write("Input Shape:", sequence.shape)

# ---------------- MODEL SELECTION ----------------
model_choice = st.radio(
    "Select Model",
    ["RNN", "LSTM", "GRU", "Compare All Models"]
)

# ---------------- PREDICT ----------------
if st.button("🚀 Predict"):

    rnn_pred = float(rnn_model.predict(sequence)[0][0])
    lstm_pred = float(lstm_model.predict(sequence)[0][0])
    gru_pred = float(gru_model.predict(sequence)[0][0])

    # ---------------- SINGLE MODEL OUTPUT ----------------
    if model_choice == "RNN":
        st.success(f"🧠 RNN Prediction: {rnn_pred:.2f}")

    elif model_choice == "LSTM":
        st.success(f"🧠 LSTM Prediction: {lstm_pred:.2f}")

    elif model_choice == "GRU":
        st.success(f"🧠 GRU Prediction: {gru_pred:.2f}")

    # ---------------- COMPARE ALL ----------------
    else:
        df = pd.DataFrame({
            "Model": ["RNN", "LSTM", "GRU"],
            "Prediction": [rnn_pred, lstm_pred, gru_pred]
        })

        st.subheader("📋 Comparison Table")
        st.dataframe(df, use_container_width=True)

        st.subheader("📈 Comparison Graph")
        st.bar_chart(df.set_index("Model"))

        best = df.loc[df["Prediction"].idxmax()]

        st.success(f"🏆 Best Model: {best['Model']}")

        st.info("LSTM/GRU usually perform better in time-series due to memory capability.")