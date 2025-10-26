import streamlit as st
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from tensorflow.keras.models import load_model

# --------------------------------------------------
# Load trained CNN model
# --------------------------------------------------
@st.cache_resource
def load_cnn_model():
    model = load_model("my_exo_model (4).keras")
    return model

model = load_cnn_model()

# Show model info
with st.expander("🧠 Model Info"):
    st.text("Model loaded successfully")
    st.text("Input shape expected by model: " + str(model.input_shape))
    st.text("Output shape: " + str(model.output_shape))
    model.summary(print_fn=lambda x: st.text(x))

# --------------------------------------------------
# Preprocessing Pipeline
# --------------------------------------------------
def preprocess_lightcurve(df):
    # Keep only numeric columns, drop index/label
    df = df.select_dtypes(include=[np.number])
    for col in df.columns:
        if "label" in col.lower() or "index" in col.lower():
            df = df.drop(columns=[col])

    st.write("🔍 Before flattening:", df.shape)
    st.write(df.head())

    # Flatten to 1D array
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

    # 3️⃣ Normalize to [-1, 1]
    minval, maxval = np.min(X), np.max(X)
    if maxval - minval != 0:
        X = (X - minval) / (maxval - minval)
        X = 2 * X - 1
    else:
        st.warning("Normalization skipped (flat signal)")

    # 4️⃣ Expand dims for CNN
    X = np.expand_dims(X, axis=2)
    X = X.astype(np.float32)  # ✅ Ensure float32 for Keras

    st.write("✅ Finished preprocessing. Stats:")
    st.write("Min:", np.min(X), "Max:", np.max(X))
    st.write("Mean:", np.mean(X), "Std:", np.std(X))

    return X

# --------------------------------------------------
# Streamlit App UI
# --------------------------------------------------
st.title("🚀 HELIOS — Exoplanet Detector")
st.markdown("""
Upload a **light curve CSV file**.  
The app applies the *same preprocessing pipeline* used during training and predicts whether
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
        st.write("Raw model output:", preds)

        confidence = float(preds[0][0])
        label = "🪐 Exoplanet" if confidence > 0.5 else "❌ Not Exoplanet"

        # Display results
        st.subheader("🔭 Prediction Result")
        st.write(f"**Prediction:** {label}")
        st.write(f"**Confidence:** {confidence:.2f}")

        # Optional: plot the light curve
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
