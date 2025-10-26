import streamlit as st
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.preprocessing import RobustScaler
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

    # 3️⃣ Normalize (min–max)
    minval, maxval = np.min(X), np.max(X)
    X = (X - minval) / (maxval - minval + 1e-8)

    # 4️⃣ Robust scaling
    scaler = RobustScaler()
    X = scaler.fit_transform(X)

    # Expand dims for CNN
    X = np.expand_dims(X, axis=2)
    return X

# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------
st.title("🪐 HELIOS — Exoplanet Detector")
st.markdown("""
Upload a **light curve CSV file** (e.g. a Kepler row sample) and this model will apply
the *same augmentation pipeline* and predict whether it likely represents an **Exoplanet**.
""")

uploaded_file = st.file_uploader("📂 Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ File uploaded successfully! Shape: {df.shape}")

        processed = preprocess_lightcurve(df)

        st.write("Processed data shape:", processed.shape)
        preds = model.predict(processed)
        confidence = float(preds[0][0])
        label = "🪐 Exoplanet" if confidence > 0.5 else "❌ Not Exoplanet"

        st.subheader("🔭 Prediction Result")
        st.write(f"**Prediction:** {label}")
        st.write(f"**Confidence:** {confidence:.4f}")

        # Drop non-flux columns for plotting
        plot_df = df.copy()
        for col in df.columns:
            if col.lower() in ["index", "label", "labels"]:
                plot_df = plot_df.drop(columns=[col])

        # Make sure it’s a 1D array for plotting
        st.line_chart(plot_df.values.flatten())

    except Exception as e:
        st.error(f"⚠️ Error: {e}")
else:
    st.info("⬆️ Please upload a CSV file to begin.")

st.markdown("---")
st.caption("Created by MaxHero123 — Powered by Streamlit + TensorFlow")

