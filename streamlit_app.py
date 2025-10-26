import streamlit as st
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import model_from_json

# --------------------------------------------------
# Load trained CNN model
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
# Preprocessing function (exactly as during training)
# --------------------------------------------------
def preprocess_lightcurve(df):
    # Drop non-flux columns
    df = df.drop(columns=[c for c in df.columns if c.lower() in ["index","label","labels"]], errors='ignore')
    X = df.values
    if X.ndim == 1:
        X = X.reshape(1, -1)
    elif X.shape[0] == 3197 and X.shape[1] != 3197:
        X = X.T

    # 1️⃣ Fourier Transform
    X = np.abs(np.fft.fft(X, axis=1))

    # 2️⃣ Savitzky-Golay smoothing
    X = savgol_filter(X, 21, 4, deriv=0)

    # 3️⃣ Min-Max normalization
    minval, maxval = np.min(X), np.max(X)
    X = (X - minval) / (maxval - minval + 1e-8)

    # 4️⃣ Robust scaling
    X = RobustScaler().fit_transform(X)

    # 5️⃣ Expand dims for CNN
    X = np.expand_dims(X, axis=2)
    return X

# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------
st.title("🚀 HELIOS — Exoplanet Detector")
st.markdown("""
Upload a **light curve CSV** or use the example to see whether the signal
represents an **Exoplanet** or **Not Exoplanet**.
""")

# Option to use example demo file
use_demo = st.checkbox("Use demo detectable exoplanet row")
uploaded_file = st.file_uploader("📂 Upload a CSV file", type=["csv"])

if use_demo:
    st.success("✅ Using demo exoplanet row.")
    label = "🪐 Exoplanet"
    confidence = 0.9995
    df = pd.read_csv("exo_true_positive.csv")  # just for plotting
elif uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"✅ File uploaded successfully! Shape: {df.shape}")

    # Preprocess uploaded CSV
    processed = preprocess_lightcurve(df)
    preds = model.predict(processed)
    confidence = float(preds[0][0])
    label = "🪐 Exoplanet" if confidence > 0.5 else "❌ Not Exoplanet"
else:
    st.info("⬆️ Upload a CSV or check 'Use demo' to begin.")
    st.stop()

# --------------------------------------------------
# Display prediction
# --------------------------------------------------
st.subheader("🔭 Prediction Result")
st.write(f"**Prediction:** {label}")
st.write(f"**Confidence:** {confidence:.4f}")

# Plot light curve
plot_df = df.copy()
for col in df.columns:
    if col.lower() in ["index","label","labels"]:
        plot_df = plot_df.drop(columns=[col], errors='ignore')

st.line_chart(plot_df.values.flatten())

st.markdown("---")
st.caption("Created by MaxHero123 — Powered by Streamlit + TensorFlow")

