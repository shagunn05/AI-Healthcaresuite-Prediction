
import numpy as np
import pandas as pd
import plotly.express as px
import os
from tensorflow.keras.models import load_model
import streamlit as st


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Healthcare Dashboard",
    page_icon="🩺",
    layout ="wide",
    
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

.big-title{
    text-align:center;
    color:#2563EB;
    font-size:42px;
    font-weight:bold;
    margin-bottom:10px;
}

.subtitle{
    text-align:center;
    color:#64748B;
    font-size:18px;
    margin-bottom:20px;
}

div[data-testid="metric-container"]{
    background-color:#F8FAFC;
    border:1px solid #E2E8F0;
    padding:15px;
    border-radius:15px;
    text-align:center;
    box-shadow:0px 2px 8px rgba(0,0,0,0.08);
}

.stButton > button{
    width:100%;
    height:55px;
    border-radius:12px;
    font-size:20px;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown(
    "<div class='big-title'>🩺 AI Healthcare Dashboard</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>AI Powered Diabetes Prediction System</div>",
    unsafe_allow_html=True
)

st.markdown("---")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_ann_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "Models", "ann_model.keras")

    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found: {MODEL_PATH}")
        st.stop()

    return load_model(MODEL_PATH)

model = load_ann_model()

# ---------------- INPUT SECTION ----------------
st.subheader("📋 Patient Information")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", 0, 20, 1)
    glucose = st.number_input("Glucose", 0, 300, 120)
    blood_pressure = st.number_input("Blood Pressure", 0, 200, 70)
    skin_thickness = st.number_input("Skin Thickness", 0, 100, 20)

with col2:
    insulin = st.number_input("Insulin", 0, 1000, 80)
    bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
    diabetes_pedigree = st.number_input(
        "Diabetes Pedigree Function",
        0.0,
        5.0,
        0.5
    )
    age = st.number_input("Age", 1, 120, 30)

st.markdown("")

# ---------------- PREDICT BUTTON ----------------
if st.button("🔍 Predict Diabetes"):

    input_data = np.array([[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        diabetes_pedigree,
        age
    ]])

    prediction = model.predict(input_data, verbose=0)
    score = float(prediction[0][0])

    # ---------------- RESULT ----------------
    st.markdown("---")
    st.subheader("🎯 Prediction Result")

    c1, c2 = st.columns(2)

    with c1:
        st.metric(
            "Prediction Score",
            f"{score:.4f}"
        )

    with c2:
        if score > 0.5:
            st.metric("Risk Status", "⚠️ High Risk")
        else:
            st.metric("Risk Status", "✅ Low Risk")

    # ---------------- RISK BAR ----------------
    st.subheader("📊 Risk Percentage")

    risk_percent = min(score * 100, 100)

    st.progress(int(risk_percent))
    st.write(f"Risk Level: {risk_percent:.2f}%")

    # ---------------- SUMMARY ----------------
    st.subheader("📈 Patient Summary")

    m1, m2, m3 = st.columns(3)

    with m1:
        st.metric("Glucose", glucose)

    with m2:
        st.metric("BMI", bmi)

    with m3:
        st.metric("Age", age)

    # ---------------- GRAPH ----------------
    st.subheader("📉 Health Parameters Analysis")

    graph_data = pd.DataFrame({
        "Feature": [
            "Pregnancies",
            "Glucose",
            "Blood Pressure",
            "Skin Thickness",
            "Insulin",
            "BMI",
            "DPF",
            "Age"
        ],
        "Value": [
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            diabetes_pedigree,
            age
        ]
    })

    fig = px.bar(
        graph_data,
        x="Feature",
        y="Value",
        title="Patient Health Parameters"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ---------------- REPORT ----------------
    st.subheader("📋 Prediction Report")

    report = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "Blood Pressure": blood_pressure,
        "Skin Thickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DPF": diabetes_pedigree,
        "Age": age,
        "Prediction Score": round(score, 4),
        "Risk Status": (
            "High Risk"
            if score > 0.5
            else "Low Risk"
        )
    }

    st.json(report)

    # ---------------- FINAL MESSAGE ----------------
    if score > 0.5:
        st.error(
            "⚠️ Patient has a higher probability of diabetes."
        )
    else:
        st.success(
            "✅ Patient has a lower probability of diabetes."
        )

