import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import model_from_json
from scipy.fft import fft
from astropy.io import fits

# ---------------------------
# Load your trained model
# ---------------------------
@st.cache_resource
def load_model():
    with open("my_exo_model_arch.json", "r") as json_file:
        loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    model.load_weights("my_exo_model.weights.h5")
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

model = load_model()
st.sidebar.success("✅ Model loaded successfully!")

# ---------------------------
# Helper functions
# ---------------------------
def load_light_curve(file):
    if file.name.endswith(".fits"):
        with fits.open(file) as hdul:
            data = hdul[1].data
            flux = data["PDCSAP_FLUX"]
            flux = flux[np.isfinite(flux)]
            return flux
    elif file.name.endswith(".csv"):
        df = pd.read_csv(file)
        flux = df.iloc[:, 1].values if df.shape[1] > 1 else df.iloc[:, 0].values
        return flux
    else:
        raise ValueError("Unsupported file format. Please upload .csv or .fits")

def preprocess_flux(flux):
    flux = np.nan_to_num(flux)
    flux = flux - np.mean(flux)
    flux = flux / np.max(np.abs(flux)) if np.max(np.abs(flux)) != 0 else flux
    fft_vals = np.abs(fft(flux))
    fft_vals = fft_vals / np.max(fft_vals) if np.max(fft_vals) != 0 else fft_vals
    fft_vals = fft_vals.reshape(1, -1, 1)
    return fft_vals

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🔭 Exoplanet Classifier")
st.write("Upload a Kepler light curve (.csv or .fits) to detect potential exoplanets.")

uploaded_file = st.file_uploader("Upload file", type=["csv", "fits"])

if uploaded_file:
    st.success("✅ File uploaded successfully!")

    try:
        flux = load_light_curve(uploaded_file)
        st.write(f"Data shape: {flux.shape}")

        processed = preprocess_flux(flux)
        st.write(f"Processed data shape: {processed.shape}")

        prediction = model.predict(processed)
        confidence = float(prediction[0][0])

        st.subheader("🔭 Prediction Result")
        if confidence >= 0.5:
            st.success(f"🪐 Exoplanet detected! (Confidence: {confidence:.2f})")
        else:
            st.error(f"❌ Not an exoplanet (Confidence: {confidence:.2f})")

    except Exception as e:
        st.error(f"Error processing file: {e}")
