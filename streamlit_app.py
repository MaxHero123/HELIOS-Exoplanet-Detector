from sklearn.preprocessing import RobustScaler

def preprocess_lightcurve(df):
    for col in df.columns:
        if col.lower() in ["index", "label", "labels"]:
            df = df.drop(columns=[col])

    X = df.values
    if X.ndim == 1:
        X = X.reshape(1, -1)
    elif X.shape[0] == 3197 and X.shape[1] != 3197:
        X = X.T

    # Fourier transform
    X = np.abs(np.fft.fft(X, axis=1))

    # Savitzky–Golay smoothing
    from scipy.signal import savgol_filter
    X = savgol_filter(X, 21, 4, deriv=0)

    # Min–max normalization
    minval, maxval = np.min(X), np.max(X)
    X = (X - minval) / (maxval - minval + 1e-8)

    # Robust scaling (fit on input itself)
    scaler = RobustScaler()
    X = scaler.fit_transform(X)

    # Expand dims for CNN
    X = np.expand_dims(X, axis=2)
    return X
