import streamlit as st
import numpy as np
import pandas as pd
import json, joblib
from scipy.signal import savgol_filter
from tensorflow.keras.models import load_model

# -------------------------------------------------------
# 📦 Constants
# -------------------------------------------------------
EXPECTED_LEN = 3197

# -------------------------------------------------------
# 🧠 Load model and preprocessing assets
# -------------------------------------------------------
@st.cache_resource
def load_all():
    model = load_model("my_exo_model.keras")

    # Load normalization parameters
    with open("norm_params.json") as f:
        params = json.load(f)
    minval, maxval = params["minval"], params["maxval"]

    # Load RobustScaler
    scaler = joblib.load("robust_scaler.joblib")

    return model, minval, maxval, scaler

model, MINVAL, MAXVAL, robust_scaler = load_all()

st.title("🪐 HELIOS — Exoplanet Detector")
st.caption("Deep Learning Exoplanet Classifier — Fourier + SavGol + RobustScaler pipeline")

# -------------------------------------------------------
# 🧩 Helper Functions
# -------------------------------------------------------
def coerce_numeric_1d(df: pd.DataFrame) -> np.ndarray:
    """Extract numeric data and flatten to 1D."""
    df = df.select_dtypes(include=["number"]).copy()
    drop = [c for c in df.columns if c.lower() in {"label", "labels", "y", "target", "class", "index", "idx"}]
    if drop:
        df = df.drop(columns=drop)

    arr = df.values
    if arr.ndim == 2:
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[1] == 1:
            arr = arr[:, 0]
        elif arr.shape[1] == EXPECTED_LEN:
            arr = arr[0, :]
        elif arr.shape[0] == EXPECTED_LEN:
            arr = arr[:, 0]
        else:
            arr = arr.ravel()
    return arr.astype("float64", copy=False)

def resample_to_len(x: np.ndarray, L: int) -> np.ndarray:
    """If user uploads non-3197-length curve, resample to length 3197."""
    if x.shape[0] == L:
        return x
    old = np.linspace(0.0, 1.0, x.shape[0])
    new = np.linspace(0.0, 1.0, L)
    return np.interp(new, old, x)

def make_model_input(df: pd.DataFrame) -> np.ndarray:
    """Apply preprocessing pipeline exactly as in training."""
    x = coerce_numeric_1d(df)
    x = resample_to_len(x, EXPECTED_LEN)

    # 1️⃣ Fourier transform magnitude
    X = np.abs(np.fft.fft(x, axis=0))

    # 2️⃣ Savitzky–Golay smoothing
    win = 21 if EXPECTED_LEN >= 21 else (EXPECTED_LEN // 2 * 2 + 1)
    X = savgol_filter(X, win, 4, deriv=0)

    # 3️⃣ Global min–max normalization (training constants)
    X = (X - MINVAL) / (MAXVAL - MINVAL + 1e-8)

    # 4️⃣ RobustScaler (fit on training set)
    X = robust_scaler.transform(X.reshape(1, -1))

    # 5️⃣ Expand dims for Conv1D
    X = X.reshape(1, EXPECTED_LEN, 1).astype("float32")

    return X

# -------------------------------------------------------
# 📤 File Upload UI
# -------------------------------------------------------
uploaded = st.file_uploader("📂 Upload a light curve CSV file", type=["csv"])

if uploaded:
    try:
        df = pd.read_csv(uploaded)
        st.success(f"✅ File uploaded! Shape: {df.shape}")

        # Plot raw flux curve
        numeric = df.select_dtypes(include=["number"]).drop(columns=[c for c in df.columns if c.lower() in {"label","labels"}], errors="ignore")
        st.line_chart(numeric.values.flatten(), height=200)

        inp = make_model_input(df)
        st.write("Processed tensor shape:", inp.shape)

        # Predict
        y = float(model.predict(inp)[0][0])
        st.metric("Exoplanet Probability", f"{y:.4f}")
        st.write("Prediction:", "🪐 Exoplanet" if y >= 0.5 else "❌ Not Exoplanet")

    except Exception as e:
        st.error(f"⚠️ Error while processing file: {e}")

else:
    st.info("⬆️ Please upload a CSV file to start analysis.")

# -------------------------------------------------------
# 🧪 Sanity Check (Optional)
# -------------------------------------------------------
st.markdown("---")
if st.button("Run Sanity Check (use built-in confirmed exoplanet sample)"):
    try:
        df_ok = pd.read_csv("exo_true_positive.csv")
        inp_ok = make_model_input(df_ok)
        p = float(model.predict(inp_ok)[0][0])
        st.success(f"✅ Sanity-check probability: {p:.4f}")
        st.write("Expected result: **🪐 Exoplanet**")
    except Exception as e:
        st.error(f"⚠️ Could not run sanity check: {e}")

st.markdown("---")
st.caption("Created by Maximilian Solomon — Powered by TensorFlow + Streamlit 🌌")

