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
    # Drop index or label columns
    for col in df.columns:
        if col.lower() in ["index", "label", "labels"]:
            df = df.drop(columns=[col])

    X = df.values

    # Ensure correct shape (1 x 3197)
    if X.shape[0] == 3197 and X.shape[1] == 1:
        X = X.T
    elif X.ndim == 1:
        X = X.reshape(1, -1)

    # 1️⃣ Fourier Transform
    X = np.abs(np.fft.fft(X, axis=1))

    # 2️⃣ Savitzky–Golay smoothing
    X = savgol_filter(X, 21, 4, deriv=0)

    # 3️⃣ Normalize (identical to training)
    minval = np.min(X)
    maxval = np.max(X)
    X = (X - minval) / (maxval - minval)

    # 4️⃣ Robust scaling
    scaler = RobustScaler()
    X = scaler.fit_transform(X)

    # 5️⃣ Expand dims for CNN
    X = np.expand_dims(X, axis=2)

    return X

# --------------------------------------------------
# Streamlit App UI
# --------------------------------------------------
st.title("🚀 HELIOS — Exoplanet Detector")
st.markdown("""
Upload a **light curve CSV file** from the Kepler dataset and this app will
apply the same augmentation pipeline used during training and predict whether
the signal likely represents an **Exoplanet** or **Not Exoplanet**.
""")

uploaded_file = st.file_uploader("📂 Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("✅ File uploaded successfully!")
        st.write(f"Data shape: {df.shape}")

        # Preprocess the light curve
        processed = preprocess_lightcurve(df)

        # Predict
        preds = model.predict(processed)
        confidence = float(np.mean(preds))
        label = "🪐 Exoplanet" if confidence > 0.5 else "❌ Not Exoplanet"

        # Display results
        st.subheader("🔭 Prediction Result")
        st.write(f"**Prediction:** {label}")
        st.write(f"**Confidence:** {confidence:.2f}")

        # Optional: visualize one sample
        st.line_chart(df.iloc[0, :])
    except Exception as e:
        st.error(f"An error occurred during processing: {e}")
else:
    st.info("⬆️ Please upload a light curve CSV file to begin.")

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.caption("Created by MaxHero123 — Powered by Streamlit + TensorFlow")
