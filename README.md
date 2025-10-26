# 🪐 HELIOS — Exoplanet Detector

**HELIOS** is a deep learning–based tool that classifies **light curves** from Kepler data to detect potential **exoplanets**.  
It uses a Convolutional Neural Network (CNN) trained on the NASA Exoplanet dataset (via Kaggle’s *exoTrain* and *exoTest* files).

---

## 🚀 Features

- Upload a light curve `.csv` file  
- Automatic preprocessing (Fourier transform, Savitzky–Golay filtering, scaling)
- CNN-based classification for exoplanet likelihood
- Confidence score output and visualization of the uploaded light curve

---

## 🧠 Model Details

The trained CNN model architecture and weights are included:
- `my_exo_model_arch.json`
- `my_exo_model.weights.h5`

These are automatically loaded in the Streamlit app at runtime.

---

## 🧩 How It Works

1. **Upload a light curve CSV** (Kepler-style time–flux data).  
2. The pipeline applies:
   - Fast Fourier Transform (FFT)  
   - Savitzky–Golay smoothing  
   - Min–max normalization  
   - Robust scaling  
3. The preprocessed signal is fed into the CNN.  
4. The model predicts a **probability** of being an exoplanet.

---

## 🧰 Tech Stack

- [Streamlit](https://streamlit.io/)
- [TensorFlow / Keras](https://www.tensorflow.org/)
- [scikit-learn](https://scikit-learn.org/)
- [NumPy](https://numpy.org/)
- [SciPy](https://scipy.org/)
- [Pandas](https://pandas.pydata.org/)

---

## ⚙️ Installation (Local Setup)

To run locally:

```bash
git clone https://github.com/MaxHero123/HELIOS-Exoplanet-Detector.git
cd HELIOS-Exoplanet-Detector
pip install -r requirements.txt
streamlit run app.py
