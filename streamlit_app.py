import numpy as np
import pandas as pd
import json
import joblib
from scipy.signal import savgol_filter
import streamlit as st
from tensorflow.keras.models import load_model

# -----------------------------
# ✅ Constants
# -----------------------------
EXPECTED_LEN = 3197
MODEL_FILE = "my_exoplanet_model.keras"  # updated filename
NORM_PARAMS_FILE = "norm_params.json"
SCALER_FILE = "robust_scaler.joblib"
POSITIVE_SAMPLE_FILE = "exo_true_positive.csv"

# -----------------------------
# ✅ Load all model artifacts
# -----------------------------
@st.cache_resource
def load_all():
    model = load_model(MODEL_FILE)
    with open(NORM_PARAMS_FILE) as f:
        params = json.load(f)
    scaler = joblib.load(SCALER_FILE)
    return model, params["minval"], params["maxval"], scaler

model, MINVAL, MAXVAL, robust_scaler = load_all()

st.title("🔭 Helios: Exoplanet Detector")
st.write("This model predicts whether a light curve likely represents an exoplanet transit signal.")
st.write("Model input shape:", model.input_shape)

# -----------------------------
# ✅ Helper functions
# -----------------------------
def coerce_numeric_1d(df: pd.DataFrame) -> np.ndarray:
    """Extracts numeric flux values from a DataFrame, dropping label or index columns."""
    df = df.select_dtypes(include=["number"]).copy()
    drop = [c for c in df.columns if c.lower() in {"label","labels","y","target","class","index","idx"}]
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
    """Resamples flux series to length L using linear interpolation."""
    if x.shape[0] == L:
        return x
    old = np.linspace(0.0, 1.0, x.shape[0])
    new = np.linspace(0.0, 1.0, L)
    return np.interp(new, old, x)


def make_model_input(df: pd.DataFrame) -> np.ndarray:
    """Mirrors the training preprocessing pipeline exactly."""
    # 1️⃣ Numeric → 3197 samples
    x = coerce_numeric_1d(df)
    x = resample_to_len(x, EXPECTED_LEN)

    # 2️⃣ FFT magnitude
    X = np.abs(np.fft.fft(x, axis=0))

    # 3️⃣ Savitzky–Golay smoothing (window=21, poly=4)
    win = 21 if EXPECTED_LEN >= 21 else (EXPECTED_LEN // 2 * 2 + 1)
    X = savgol_filter(X, win, 4, deriv=0)

    # 4️⃣ Global min–max normalization using training constants
    X = (X - MINVAL) / (MAXVAL - MINVAL + 1e-8)

    # 5️⃣ RobustScaler (same one used in training)
    X = robust_scaler.transform(X.reshape(1, -1))

    # 6️⃣ Expand dimensions → (1, 3197, 1)
    X = X.reshape(1, EXPECTED_LEN, 1).astype("float32")

    assert X.shape == (1, EXPECTED_LEN, 1), f"Bad shape {X.shape}"
    assert np.isfinite(X).all(), "NaN or inf in preprocessed input"
    return X


# -----------------------------
# ✅ UI for file upload
# -----------------------------
uploaded = st.file_uploader("📁 Upload a single light-curve CSV", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    inp = make_model_input(df)

    st.write("Tensor → model:", inp.shape, inp.dtype,
             "min:", float(inp.min()), "max:", float(inp.max()))

    # Run prediction
    y = float(model.predict(inp)[0][0])
    st.metric("Exoplanet probability", f"{y:.4f}")

    thr = st.slider("Decision threshold", 0.0, 1.0, 0.5, 0.01)
    st.write("Prediction:", "✅ Exoplanet" if y >= thr else "❌ Not Exoplanet")

# -----------------------------
# ✅ Sanity check button
# -----------------------------
if st.button("Run Sanity Check (bundled positive)"):
    df_ok = pd.read_csv(POSITIVE_SAMPLE_FILE)
    inp_ok = make_model_input(df_ok)
    p = float(model.predict(inp_ok)[0][0])
    st.success(f"Sanity-check probability: {p:.4f}")

# -----------------------------
# ✅ Optional visualization
# -----------------------------
if uploaded:
    st.subheader("🧠 Preprocessing Visualization")
    import matplotlib.pyplot as plt

    raw_flux = coerce_numeric_1d(df)
    proc_flux = np.abs(np.fft.fft(raw_flux))
    proc_flux = savgol_filter(proc_flux, 21, 4)

    fig, ax = plt.subplots(2, 1, figsize=(10, 6))
    ax[0].plot(raw_flux, color="gray")
    ax[0].set_title("Raw Input Flux")
    ax[1].plot(proc_flux, color="orange")
    ax[1].set_title("Processed (FFT + SavGol) Flux")
    st.pyplot(fig)
