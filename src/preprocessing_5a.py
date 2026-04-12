import numpy as np

import load_data

STORE_ID = "CA_1"
CAT_ID = "FOODS"
DEPT_ID = "FOODS_1"
INPUT_WINDOW = 28
TARGET_WINDOW = 7
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15

# 5A bruker alle serier i samme subset som tidligere milepæler.
subset = load_data.sales[
    (load_data.sales["store_id"] == STORE_ID)
    & (load_data.sales["cat_id"] == CAT_ID)
    & (load_data.sales["dept_id"] == DEPT_ID)
].copy()

day_cols = [col for col in subset.columns if col.startswith("d_")]
series_matrix = subset[day_cols].to_numpy(dtype=np.float32)
series_ids = subset["id"].tolist()


def split_time_axis(series_data, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO):
    """Splitter tidsaksen først for å få en strengere train/val/test-protokoll."""
    num_days = series_data.shape[1]
    train_end = int(num_days * train_ratio)
    val_end = int(num_days * (train_ratio + val_ratio))

    train_series = series_data[:, :train_end]
    val_series = series_data[:, train_end:val_end]
    test_series = series_data[:, val_end:]

    return train_series, val_series, test_series


def create_windows_for_split(series_split, input_window=INPUT_WINDOW, target_window=TARGET_WINDOW):
    """Lager windows separat innen hver split uten å krysse tidsgrenser."""
    total_window = input_window + target_window
    split_length = series_split.shape[1]

    if split_length < total_window:
        return (
            np.empty((0, input_window), dtype=np.float32),
            np.empty((0, target_window), dtype=np.float32),
        )

    windows = np.lib.stride_tricks.sliding_window_view(
        series_split, window_shape=total_window, axis=1
    )
    X = windows[:, :, :input_window]
    y = windows[:, :, input_window:]

    X = X.reshape(-1, input_window).astype(np.float32)
    y = y.reshape(-1, target_window).astype(np.float32)
    return X, y


train_series, val_series, test_series = split_time_axis(series_matrix)
X_train, y_train = create_windows_for_split(train_series)
X_val, y_val = create_windows_for_split(val_series)
X_test, y_test = create_windows_for_split(test_series)


if __name__ == "__main__":
    print("Antall serier i subset:", len(series_ids))
    print("Series matrix shape:", series_matrix.shape)
    print("Train series shape:", train_series.shape)
    print("Val series shape:", val_series.shape)
    print("Test series shape:", test_series.shape)
    print("X_train shape:", X_train.shape, "y_train shape:", y_train.shape)
    print("X_val shape:", X_val.shape, "y_val shape:", y_val.shape)
    print("X_test shape:", X_test.shape, "y_test shape:", y_test.shape)
