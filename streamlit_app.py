import streamlit as st
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from tensorflow.keras.models import model_from_json

# --------------------------------------------------
# Load model architecture and weights
# --------------------------------------------------
@st.cache_resource
def load_cnn_model():
    with open("my_exo_model_arch.json", "r") as f:
        model_json = f.read()
    model = model_from_json(model_json)
    model.load_weights("my_exo_model.weights.h5")
    return model

model = load_cnn_model()

# --------------------------------------------------
# Preprocessing (exactly as during training)
# --------------------------------------------------
def preprocess_lightcurve(df):
    # Drop label/index columns if they exist
    for col in df.columns:
        if col.lower() in ["index", "label", "labels"]:
            df = df.drop(columns=[col])

    X = df.values
    if X.ndim == 1:
        X = X.reshape(1, -1)
    elif X.shape[0] == 3197 and X.shape[1] != 3197:
        X = X.T

    # 1️⃣ Fourier transform
    X = np.abs(np.fft.fft(X, axis=1))

    # 2️⃣ Savitzky–Golay smoothing
    X = savgol_filter(X, 21, 4, deriv=0)

    # 3️⃣ Min–max normalization
    minval, maxval = np.min(X), np.max(X)
    X = (X - minval) / (maxval - minval + 1e-8)

    # Expand dims for CNN
    X = np.expand_dims(X, axis=2)
    return X

# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------
st.title("🪐 HELIOS — Exoplanet Detector")
st.markdown("""
Upload a **light curve CSV file** (e.g. a Kepler sample).  
The app will verify its shape and ask whether it has already been preprocessed.
""")

uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ File uploaded successfully! Shape: {df.shape}")

        # Check for flux length (must be 3197)
        flux_length = df.shape[0] if df.shape[0] == 3197 else df.shape[1]
        if flux_length != 3197:
            st.error("⚠️ Data must contain exactly 3197 flux points. Please upload a valid Kepler-like light curve.")
        else:
            preprocess_needed = st.radio(
                "Has this data already been preprocessed using the official HELIOS method?",
                ["No", "Yes"]
            )

            if preprocess_needed == "No":
                st.info("🔧 Running preprocessing pipeline...")
                processed = preprocess_lightcurve(df)
            else:
                st.info("✅ Using your uploaded data directly.")
                processed = df.values
                if processed.ndim == 1:
                    processed = processed.reshape(1, -1)
                processed = np.expand_dims(processed, axis=2)

            st.write("Processed data shape:", processed.shape)
            preds = model.predict(processed)
            confidence = float(preds[0][0])
            label = "🪐 Exoplanet" if confidence > 0.5 else "❌ Not Exoplanet"

            st.subheader("🔭 Prediction Result")
            st.write(f"**Prediction:** {label}")
            st.write(f"**Confidence:** {confidence:.4f}")

            st.line_chart(df.values.flatten())

    except Exception as e:
        st.error(f"⚠️ Error: {e}")
else:
    st.info("⬆️ Please upload a CSV file to begin.")

st.markdown("---")
st.caption("Created by MaxHero123 — Powered by Streamlit + TensorFlow")
