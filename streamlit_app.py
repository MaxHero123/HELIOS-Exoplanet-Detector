import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------
# Load your trained model
# ------------------------------------------------------
@st.cache_resource
def load_exoplanet_model():
    model = load_model("my_exo_model (3).keras")
    return model

model = load_exoplanet_model()

# ------------------------------------------------------
# Page Setup
# ------------------------------------------------------
st.set_page_config(page_title="HELIOS Exoplanet Detector", layout="centered")
st.title("🪐 HELIOS Exoplanet Detector")
st.write("Upload a Kepler light curve CSV and let the model predict if it contains an exoplanet!")

# ------------------------------------------------------
# File Upload
# ------------------------------------------------------
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        # Read CSV
        df = pd.read_csv(uploaded_file)
        st.subheader("Preview of Uploaded Data")
        st.dataframe(df.head())

        # ------------------------------------------------------
        # Preprocessing (match the model's training)
        # ------------------------------------------------------
        # Assume 'flux' columns are features, and we drop label/index if they exist
        X = df.filter(regex='flux|^f[0-9]+$', axis=1).values
        if X.shape[1] == 0:
            X = df.iloc[:, :-1].values  # fallback: all but last column

        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Reshape for 1D CNN
        X_scaled = np.expand_dims(X_scaled, axis=2)

        # ------------------------------------------------------
        # Prediction
        # ------------------------------------------------------
        predictions = model.predict(X_scaled)
        confidence = float(np.mean(predictions))
        label = "Exoplanet" if confidence >= 0.5 else "Not Exoplanet"

        st.subheader("🔮 Prediction Result")
        st.write(f"**Prediction:** {label}")
        st.write(f"**Confidence:** {confidence:.2f}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please upload a CSV file to begin.")
