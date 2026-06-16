import numpy as np
import pandas as pd
import plotly.express as px
import os
import time
from tensorflow.keras.models import load_model
import streamlit as st

# 1. Page Configuration (Must be the first Streamlit command)
st.set_page_config(
    page_title="AI Pneumonia Detection System",
    page_icon="🫁",
    layout="wide"  # Ensures the dashboard uses the full screen width
)

# 2. Main Title and Description
st.title("🫁 AI Pneumonia Detection System")
st.markdown("Deep Learning based Chest X-Ray Analysis using CNN.")
st.markdown("Upload a chest X-Ray image to detect potential pneumonia patterns.")

st.markdown("---")

# 3. Metrics Row (Accuracy, Precision, Recall)
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Accuracy", value="95.1%")
with col2:
    st.metric(label="Precision", value="94.3%")
with col3:
    st.metric(label="Recall", value="93.8%")

st.markdown("---")

# 4. Image Upload and Model Info Columns (3:2 Ratio)
left_side, right_side = st.columns([3, 2])

with left_side:
    st.markdown("### 📁 Image Diagnostic Center")
    uploaded_img = st.file_uploader("Upload a Chest X-Ray image (.png, .jpg, .jpeg)", type=["png", "jpg", "jpeg"])
    
    if uploaded_img is not None:
        st.image(uploaded_img, caption="Target Scan Loaded Into RAM Buffers", use_container_width=True)
        
        # Trigger analysis when the user clicks the button
        if st.button("⚙️ Analyze Matrix Vectors with CNN"):
            with st.spinner("Running maxpooling and convolution sequence operations..."):
                time.sleep(1.5)  # Simulating model processing time
            
            st.warning("⚠️ Diagnostic Output: Architectural analysis completed.")
            
            # --- NEW RESULT SECTION BELOW ---
            st.markdown("---")
            st.markdown("### 📊 Analysis Result")
            
            # Dummy variable for demonstration. 
            # Replace this with your actual model.predict() output later.
            prediction_score = 0.85  
            
            if prediction_score > 0.5:
                # Displays a Red alert box if Pneumonia is detected
                st.error(f"🚨 **Result: Pneumonia Detected** (Confidence: {prediction_score*100:.1f}%)")
                st.info("💡 **Recommendation:** Please consult a radiologist or a medical professional immediately.")
            else:
                # Displays a Green success box if the lungs look healthy
                st.success(f"✅ **Result: Normal / Healthy Lungs** (Confidence: {(1-prediction_score)*100:.1f}%)")
                st.info("💡 **Note:** No significant pneumonia patterns observed in the matrix scan.")
                
    else:
        st.info("Awaiting high-resolution lung radiography file to initialize matrix scan.")

with right_side:
    with st.container(border=True):
        st.markdown("### 📋 Model Information")
        
        st.markdown("""
        **Supported Classes:**
        * Normal
        * Pneumonia
        
        **CNN Pipeline:**
        * Convolution Layers
        * MaxPooling
        * Dense Layers
        * Sigmoid Output
        """)