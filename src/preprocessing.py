import numpy as np
import load_data

STORE_ID = "CA_1"
CAT_ID = "FOODS"
DEPT_ID = "FOODS_1"
INPUT_WINDOW = 28
TARGET_WINDOW = 7

# Filtrer salg for en spesifikk butikk og kategori.
subset = load_data.sales[
    (load_data.sales["store_id"] == STORE_ID)
    & (load_data.sales["cat_id"] == CAT_ID)
    & (load_data.sales["dept_id"] == DEPT_ID)
].copy()

# Hent ut alle dag-kolonner fra M5-datasettet.
day_cols = [col for col in subset.columns if col.startswith("d_")]

# Milepæl 2 bygger videre på én tidsserie for å få treningspipen på plass.
sample_series = subset.iloc[0][day_cols].to_numpy(dtype=np.float32)

def create_windows(series, input_window=INPUT_WINDOW, target_window=TARGET_WINDOW):
    """Lager supervised læringsvinduer fra en tidsserie."""
    series = np.asarray(series, dtype=np.float32)
    X, y = [], []

    for i in range(len(series) - input_window - target_window + 1):
        x_window = series[i : i + input_window]
        y_window = series[i + input_window : i + input_window + target_window]
        X.append(x_window)
        y.append(y_window)

    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)

def train_val_test_split(X, y, train_ratio=0.7, val_ratio=0.15):
    """Splitter vinduene kronologisk i train/val/test."""
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    X_train = X[:train_end]
    y_train = y[:train_end]

    X_val = X[train_end:val_end]
    y_val = y[train_end:val_end]

    X_test = X[val_end:]
    y_test = y[val_end:]

    return X_train, y_train, X_val, y_val, X_test, y_test

X, y = create_windows(sample_series)
X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X, y)

if __name__ == "__main__":
    print("X shape:", X.shape, "y shape:", y.shape)
    print("Treningsdata:", X_train.shape, y_train.shape)
    print("Valideringsdata:", X_val.shape, y_val.shape)
    print("Testdata:", X_test.shape, y_test.shape)
