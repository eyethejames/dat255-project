import load_data  # importerer load_data for å få tilgang til subset og day_cols

# filtrere salg for en spesifikk butikk og kategori
subset = load_data.sales [
    (load_data.sales["store_id"] == "CA_1") &
    (load_data.sales["cat_id"] == "FOODS") & # redundant since dept_id is more specific, but just to be sure
    (load_data.sales["dept_id"] == "FOODS_1")
    ].copy()

# henter ut dag-kolonnene
day_cols = [col for col in subset.columns if col.startswith("d_")]

# sjekk en faktisk tidsserie
sample_series = subset.iloc[0][day_cols].astype(int)  # og konverterer til int for enklere analyse

# etablerer sliding windows for å lage treningsdata
def create_windows(series, input_window=28, target_window=7):
    X, y = [], []

    for i in range(len(series) - input_window - target_window + 1):
        x_window = (series[i:i+input_window].values)
        y_window = (series[i+input_window:i+input_window+target_window].values)

        X.append(x_window)
        y.append(y_window)

    return X, y

X, y = create_windows(sample_series)

## en enkel split funksjon
def train_val_test_split(X, y, train_ratio=0.7, val_ratio=0.15):
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

X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X, y)

if __name__ == "__main__":
    print("Treningsdata: inputs X:", len(X_train), " Target y:", len(y_train))
    print("Valideringsdata: inputs X:", len(X_val), " Target y:", len(y_val))
    print("Testdata: inputs X:", len(X_test), " Target y:", len(y_test))