import streamlit as st
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import load_model

@st.cache_resource
def load_cnn_model():
    return load_model("my_exo_model (3).keras")

model = load_cnn_model()

# --------------------------------------------------
# Preprocessing Pipeline
# --------------------------------------------------
def preprocess_lightcurve(df):
    # Drop non-numeric and label/index columns
    df = df.select_dtypes(include=[np.number])
    for col in df.columns:
        if "label" in col.lower() or "index" in col.lower():
            df = df.drop(columns=[col])

    X = df.values

    # Handle shape
    if X.shape[0] == 3197 and X.shape[1] == 1:
        X = X.T
    elif X.ndim == 1:
        X = X.reshape(1, -1)
    elif X.shape[1] == 3198:
        X = X[:, :3197]  # trim to match training size

    st.write("📊 Shape before processing:", X.shape)

    # 1️⃣ FFT
    X = np.abs(np.fft.fft(X, axis=1))

    # 2️⃣ Savitzky–Golay smoothing
    try:
        X = savgol_filter(X, 21, 4, deriv=0)
    except Exception as e:
        st.warning(f"Savitzky-Golay failed: {e}")

    # 3️⃣ Normalize (avoid divide-by-zero)
    minval, maxval = np.min(X), np.max(X)
    if maxval != minval:
        X = (X - minval) / (maxval - minval)
    else:
        st.warning("Normalization skipped: constant signal detected")

    # 4️⃣ Robust scaling
    try:
        scaler = RobustScaler()
        X = scaler.fit_transform(X)
    except Exception as e:
        st.warning(f"Scaling failed: {e}")

    # 5️⃣ Expand dims
    X = np.expand_dims(X, axis=2)

    st.write("✅ Finished preprocessing. Stats:")
    st.write("Min:", np.min(X), "Max:", np.max(X))
    st.write("Mean:", np.mean(X), "Std:", np.std(X))

    return X

# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------
st.title("🚀 HELIOS — Exoplanet Detector")

uploaded_file = st.file_uploader("📂 Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("✅ File uploaded successfully!")
        st.write(f"Data shape: {df.shape}")

        processed = preprocess_lightcurve(df)
        st.write("Processed data shape:", processed.shape)

        preds = model.predict(processed)
        st.write("Raw model output:", preds[:10])

        confidence = float(np.mean(preds))
        label = "🪐 Exoplanet" if confidence > 0.5 else "❌ Not Exoplanet"

        st.subheader("🔭 Prediction Result")
        st.write(f"**Prediction:** {label}")
        st.write(f"**Confidence:** {confidence:.2f}")

    except Exception as e:
        st.error(f"An error occurred during processing: {e}")
else:
    st.info("⬆️ Please upload a light curve CSV file to begin.")

st.markdown("---")
st.caption("Created by MaxHero123 — Powered by Streamlit + TensorFlow")

