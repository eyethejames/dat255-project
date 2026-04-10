import numpy as np
import preprocessing


def moving_window_baseline(X_windows, forecast_horizon=7):
    """Bruker de siste observasjonene i input-vinduet som 7-dagers forecast."""
    X_windows = np.asarray(X_windows, dtype=np.float32)
    return X_windows[:, -forecast_horizon:]


def mean_absolute_error(y_pred, y_true):
    y_pred = np.asarray(y_pred, dtype=np.float32)
    y_true = np.asarray(y_true, dtype=np.float32)
    return float(np.mean(np.abs(y_pred - y_true)))


def inventory_simulation(y_pred, y_true, holding_cost=1.0, stockout_cost=5.0):
    """Regner ut enkle beslutningsmetrikker fra forecast og virkelig etterspørsel."""
    y_pred = np.asarray(y_pred, dtype=np.float32)
    y_true = np.asarray(y_true, dtype=np.float32)

    total_holding_cost = 0
    total_stockout_cost = 0
    total_demand = 0
    total_fulfilled = 0
    stockout_days = 0
    total_days = 0

    for pred_window, true_window in zip(y_pred, y_true):
        for pred, true in zip(pred_window, true_window):
            total_days += 1
            total_demand += true

            fulfilled = min(pred, true)
            total_fulfilled += fulfilled

            if pred > true: # hvis prediksjonen er høyere enn etterspørselen -> overskudd
                total_holding_cost += (pred - true) * holding_cost
            elif true > pred: # hvis etterspørsel > prediksjon -> underskudd
                total_stockout_cost += (true - pred) * stockout_cost
                stockout_days += 1

    total_cost = total_holding_cost + total_stockout_cost
    stockout_rate = stockout_days / total_days if total_days > 0 else 0
    fill_rate = total_fulfilled / total_demand if total_demand > 0 else 0

    return total_cost, stockout_rate, fill_rate


if __name__ == "__main__":
    y_pred = moving_window_baseline(preprocessing.X_test, preprocessing.TARGET_WINDOW)
    y_true = preprocessing.y_test
    mae = mean_absolute_error(y_pred, y_true)

    # Prediksjon og target, og MAE med tre desimaler
    print("Første prediksjon:", y_pred[0])
    print("Første faktiske target:", y_true[0])
    print("Mean Absolute Error (MAE):", f"{mae:.3f}")

    total_cost, stockout_rate, fill_rate = inventory_simulation(y_pred, y_true)
    print("Total Cost: {:.2f}".format(total_cost))
    print("Stockout Rate: {:.3f}".format(stockout_rate))
    print("Fill Rate: {:.3f}".format(fill_rate))
