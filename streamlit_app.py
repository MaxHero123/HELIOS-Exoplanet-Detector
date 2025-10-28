import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# --------------------------------------------------
# Load model and preprocessing assets
# --------------------------------------------------
@st.cache_resource
def load_all():
    model = load_model("my_exoplanet_model.keras")  # Updated model name
    try:
        scaler = joblib.load("robust_scaler.pkl")
    except FileNotFoundError:
        scaler = None
    MINVAL, MAXVAL = 0, 1
    return model, MINVAL, MAXVAL, scaler

model, MINVAL, MAXVAL, robust_scaler = load_all()

# --------------------------------------------------
# HELIOS Interface
# --------------------------------------------------
st.markdown(
    "<h1 style='text-align: center; color: gold;'>🔭 HELIOS: High-Accuracy Exoplanet Detector</h1>",
    unsafe_allow_html=True,
)
st.caption("A Novel ML Pipeline for High Accuracy Exoplanet Detection via Light-curve Interpretation with Optimized Fourier Analysis and SMOTE Synthesis.")
st.write("---")

st.write("### Upload your light curve CSV file below:")
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Handle missing values
    df = df.dropna(axis=1, how="all").dropna()
    flux = df.values.flatten()

    st.write("#### 📊 Uploaded Data Overview")
    st.line_chart(flux)

    # --------------------------------------------------
    # Check for correct length
    # --------------------------------------------------
    if len(flux) != 3197:
        st.error("⚠️ Data must contain exactly 3197 flux points. Please upload a valid Kepler-like light curve.")
    else:
        # Ask user if preprocessed
        preprocessed = st.radio(
            "Has your data been preprocessed using the HELIOS Fourier + normalization pipeline?",
            ("No", "Yes"),
        )

        if st.button("🚀 Run HELIOS Detection"):
            with st.spinner("Analyzing light curve data... please wait 🌠"):
                data = flux.copy()

                # Apply preprocessing only if needed
                if preprocessed == "No":
                    data = (data - np.min(data)) / (np.max(data) - np.min(data))  # normalize 0–1
                    data = np.nan_to_num(data)
                
                # Reshape for model
                data = np.expand_dims(data, axis=(0, -1))

                # Run prediction
                prediction = model.predict(data)[0][0]
                confidence = float(prediction)

            # --------------------------------------------------
            # Display Result
            # --------------------------------------------------
            if confidence >= 0.5:
                st.success(f"🌌 **Signal Detected!** HELIOS confirms this is an exoplanet with {confidence:.2%} confidence.")
                st.balloons()
            else:
                st.warning(f"❌ No exoplanet signal detected. Confidence: {confidence:.2%}")

            # --------------------------------------------------
            # Optional: Plot light curve again with label
            # --------------------------------------------------
            st.write("### 🪐 Light Curve Visualization")
            fig, ax = plt.subplots()
            ax.plot(flux, color="gold")
            ax.set_xlabel("Time (Kepler Cadence)")
            ax.set_ylabel("Flux")
            ax.set_title("Light Curve")
            st.pyplot(fig)

