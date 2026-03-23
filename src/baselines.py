import load_data  # importerer load_data for å få tilgang til subset og day_cols
import preprocessing  # importerer preprocessing for å få tilgang til X_test og y_test
import numpy as np


y_pred = [x_window[-7:] for x_window in preprocessing.X_test]  # bruker de siste 7 dagene i input som prediksjon
y_true = preprocessing.y_test

y_pred = np.array(y_pred)
y_true = np.array(y_true)

mae = np.mean(np.abs(y_pred - y_true))

if "__main__" == __name__:
    print("Første prediksjon:", y_pred[0])
    print("Første faktiske target:", y_true[0])
    print("Mean Absolute Error (MAE):", mae)

# Inventory simulation: Bruker y_pred og y_true til å regne ut total cost, stockout rate og fill rate
def inventory_simulation(y_pred, y_true, holding_cost=1.0, stockout_cost=5.0):
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
        stockout_rate = stockout_days / total_days
        fill_rate = total_fulfilled / total_demand if total_demand > 0 else 0

    return total_cost, stockout_rate, fill_rate

total_cost, stockout_rate, fill_rate = inventory_simulation(y_pred, y_true)

# skriv ut metrics med 3 desimaler
if "__main__" == __name__:
    print("Total Cost: {:.2f}".format(total_cost))
    print("Stockout Rate: {:.3f}".format(stockout_rate))
    print("Fill Rate: {:.3f}".format(fill_rate))

# sanity check av første vindu
first_pred = y_pred[:1]
first_true = y_true[:1]

total_cost, stockout_rate, fill_rate = inventory_simulation(first_pred, first_true)
if "__main__" == __name__:
    print("First window total cost: {:.2f}".format(total_cost))
    print("First window stockout rate: {:.3f}".format(stockout_rate))
    print("First window fill rate: {:.3f}".format(fill_rate))

# sanity check av tilfeldig vindu
idx = 67
random_pred = y_pred[idx:idx+1]
random_true = y_true[idx:idx+1]

if "__main__" == __name__:
    print("Tilfeldig prediksjon:", random_pred[0])
    print("Tilfeldig faktiske target:", random_true[0])

    mae_random = np.mean(np.abs(random_pred - random_true))

total_cost, stockout_rate, fill_rate = inventory_simulation(random_pred, random_true)

if "__main__" == __name__:
    print("Mean Absolute Error (MAE):", mae_random)
    print("Random window total cost: {:.2f}".format(total_cost))
    print("Random window stockout rate: {:.3f}".format(stockout_rate))
    print("Random window fill rate: {:.3f}".format(fill_rate))