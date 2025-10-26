import streamlit as st
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import load_model

# --------------------------------------------------
# Load trained CNN model
# --------------------------------------------------
@st.cache_resource
def load_cnn_model():
    model = load_model("my_exo_model (3).keras")
    return model

model = load_cnn_model()

# --------------------------------------------------
# Preprocessing Pipeline (must match training exactly)
# --------------------------------------------------
def preprocess_lightcurve(df):
    df = df.select_dtypes(include=[np.number])
    for col in df.columns:
        if "label" in col.lower() or "index" in col.lower():
            df = df.drop(columns=[col])

    st.write("🔍 Before flattening:", df.shape)

    if df.shape[0] == 1:
        X = df.values
    else:
        X = df.values.flatten().reshape(1, -1)
    st.write("📊 Shape before processing:", X.shape)

    # 1️⃣ FFT
    X = np.abs(np.fft.fft(X, axis=1))
    st.write("After FFT — Min:", np.min(X), "Max:", np.max(X))

    # 2️⃣ Savitzky–Golay smoothing
    try:
        X = savgol_filter(X, 21, 4, deriv=0)
    except Exception as e:
        st.warning(f"Savitzky–Golay failed: {e}")

    # ✅ Simple robust normalization instead of scaler
    minval = np.min(X)
    maxval = np.max(X)
    if maxval - minval != 0:
        X = (X - minval) / (maxval - minval)
    else:
        st.warning("Normalization skipped (flat signal).")

    # Optionally shift range to -1..1
    X = 2 * X - 1

    # 3️⃣ Expand dims for CNN
    X = np.expand_dims(X, axis=2)

    st.write("✅ Finished preprocessing. Stats:")
    st.write("Min:", np.min(X), "Max:", np.max(X))
    st.write("Mean:", np.mean(X), "Std:", np.std(X))

    return X

# --------------------------------------------------
# Streamlit App UI
# --------------------------------------------------
st.title("🚀 HELIOS — Exoplanet Detector")

st.markdown("""
Upload a **light curve CSV file** from the Kepler dataset.  
This app applies the *exact same preprocessing pipeline* used during training  
and predicts whether the signal is likely an **Exoplanet** or **Not Exoplanet**.
""")

uploaded_file = st.file_uploader("📂 Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("✅ File uploaded successfully!")
        st.write(f"Data shape: {df.shape}")

        # Preprocess
        processed = preprocess_lightcurve(df)
        st.write("Processed data shape:", processed.shape)

        # Predict
        preds = model.predict(processed)
        st.write("Raw model output:", preds)

        confidence = float(np.mean(preds))
        label = "🪐 Exoplanet" if confidence > 0.5 else "❌ Not Exoplanet"

        # Display
        st.subheader("🔭 Prediction Result")
        st.write(f"**Prediction:** {label}")
        st.write(f"**Confidence:** {confidence:.2f}")

        # Optional: visualize uploaded curve
        st.line_chart(df.select_dtypes(include=[np.number]).iloc[:, 0])

    except Exception as e:
        st.error(f"An error occurred during processing: {e}")

else:
    st.info("⬆️ Please upload a light curve CSV file to begin.")

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.caption("Created by MaxHero123 — Powered by Streamlit + TensorFlow")

