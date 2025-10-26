def preprocess_lightcurve(df):
    # Drop non-numeric and label/index columns
    df = df.select_dtypes(include=[np.number])
    for col in df.columns:
        if "label" in col.lower() or "index" in col.lower():
            df = df.drop(columns=[col])

    # Flatten to 1D array (3197,)
    X = df.values.flatten()

    # Ensure shape (1, 3197)
    X = X.reshape(1, -1)
    st.write("📊 Shape before processing:", X.shape)

    # 1️⃣ FFT
    X = np.abs(np.fft.fft(X, axis=1))

    # 2️⃣ Savitzky–Golay smoothing
    try:
        X = savgol_filter(X, 21, 4, deriv=0)
    except Exception as e:
        st.warning(f"Savitzky–Golay failed: {e}")

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

