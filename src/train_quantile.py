import numpy as np
import torch
import torch.nn as nn

import preprocessing
from baselines import inventory_simulation, mean_absolute_error, moving_window_baseline
from models.tcn import QuantileTCN, SimpleTCN
from training_utils import build_dataloader, predict, set_seed, to_tensor, train_model

BATCH_SIZE = 32
EPOCHS = 60
POINT_LEARNING_RATE = 1e-3
QUANTILE_LEARNING_RATE = 5e-4
SEED = 42
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MIN_DELTA = 1e-4
QUANTILES = (0.1, 0.5, 0.9)
ORDER_UP_TO_QUANTILE = 0.9


class PinballLoss(nn.Module):
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = tuple(quantiles)

    def forward(self, predictions, targets):
        quantiles = predictions.new_tensor(self.quantiles).view(1, -1, 1)
        targets = targets.unsqueeze(1)
        errors = targets - predictions
        loss = torch.maximum(quantiles * errors, (quantiles - 1.0) * errors)
        return loss.mean()


def pinball_loss_np(predictions, targets, quantiles):
    predictions = np.asarray(predictions, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)[:, None, :]
    quantiles = np.asarray(quantiles, dtype=np.float32).reshape(1, -1, 1)
    errors = targets - predictions
    losses = np.maximum(quantiles * errors, (quantiles - 1.0) * errors)
    return float(np.mean(losses))


def quantile_coverages(predictions, targets, quantiles):
    coverages = {}
    for index, quantile in enumerate(quantiles):
        coverage = np.mean(targets <= predictions[:, index, :])
        coverages[quantile] = float(coverage)
    return coverages


def interval_coverage(predictions, targets, lower_quantile=0.1, upper_quantile=0.9):
    lower_index = QUANTILES.index(lower_quantile)
    upper_index = QUANTILES.index(upper_quantile)
    inside = (targets >= predictions[:, lower_index, :]) & (
        targets <= predictions[:, upper_index, :]
    )
    return float(np.mean(inside))


def extract_quantile(predictions, quantile):
    quantile_index = QUANTILES.index(quantile)
    return predictions[:, quantile_index, :]


def prepare_inventory_forecast(predictions):
    return np.clip(np.rint(predictions), 0, None)


def evaluate_quantile_policies(quantile_predictions, targets, quantiles):
    policy_results = {}

    for quantile in quantiles:
        policy_forecast = prepare_inventory_forecast(
            extract_quantile(quantile_predictions, quantile)
        )
        policy_results[quantile] = {
            "forecast": policy_forecast,
            "mae": mean_absolute_error(policy_forecast, targets),
            "inventory": inventory_simulation(policy_forecast, targets),
        }

    return policy_results


def run_point_forecast(train_loader, val_loader, test_loader, device):
    set_seed(SEED)
    model = SimpleTCN(
        input_length=preprocessing.INPUT_WINDOW,
        output_size=preprocessing.TARGET_WINDOW,
        hidden_channels=32,
    ).to(device)
    loss_fn = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=POINT_LEARNING_RATE)

    training_result = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        epochs=EPOCHS,
        patience=EARLY_STOPPING_PATIENCE,
        min_delta=EARLY_STOPPING_MIN_DELTA,
    )
    predictions = predict(model, test_loader, device)

    return {
        "predictions": predictions,
        "training_result": training_result,
    }


def run_quantile_forecast(train_loader, val_loader, test_loader, device):
    set_seed(SEED)
    model = QuantileTCN(
        input_length=preprocessing.INPUT_WINDOW,
        output_size=preprocessing.TARGET_WINDOW,
        hidden_channels=32,
        num_quantiles=len(QUANTILES),
    ).to(device)
    loss_fn = PinballLoss(QUANTILES)
    optimizer = torch.optim.Adam(model.parameters(), lr=QUANTILE_LEARNING_RATE)

    training_result = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        epochs=EPOCHS,
        patience=EARLY_STOPPING_PATIENCE,
        min_delta=EARLY_STOPPING_MIN_DELTA,
    )
    predictions = predict(model, test_loader, device)

    return {
        "predictions": predictions,
        "training_result": training_result,
    }


def print_training_summary(label, training_result):
    if training_result["stopped_early"]:
        print(
            f"{label} stoppet tidlig ved epoch {training_result['stopped_epoch']} "
            f"av maks {EPOCHS}."
        )
    else:
        print(f"{label} fullførte alle {EPOCHS} epochene.")

    print(
        f"{label} bruker beste valideringsmodell fra epoch "
        f"{training_result['best_epoch']}/{EPOCHS} "
        f"med val-loss {training_result['best_val_loss']:.4f}."
    )


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train = to_tensor(preprocessing.X_train, add_channel_dim=True)
    y_train = to_tensor(preprocessing.y_train)
    X_val = to_tensor(preprocessing.X_val, add_channel_dim=True)
    y_val = to_tensor(preprocessing.y_val)
    X_test = to_tensor(preprocessing.X_test, add_channel_dim=True)
    y_test = to_tensor(preprocessing.y_test)
    test_targets = preprocessing.y_test

    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_val shape:", X_val.shape)
    print("y_val shape:", y_val.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)
    print("Device:", device)
    print("Quantiles:", QUANTILES)
    print("Quantile output shape:", "(batch, 3, 7)")

    train_loader = build_dataloader(X_train, y_train, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = build_dataloader(X_val, y_val, batch_size=BATCH_SIZE)
    test_loader = build_dataloader(X_test, y_test, batch_size=BATCH_SIZE)

    print("\nPoint forecast TCN:")
    point_result = run_point_forecast(train_loader, val_loader, test_loader, device)

    print("\nQuantile forecast TCN:")
    quantile_result = run_quantile_forecast(train_loader, val_loader, test_loader, device)

    baseline_predictions = moving_window_baseline(
        preprocessing.X_test, forecast_horizon=preprocessing.TARGET_WINDOW
    )

    point_predictions = point_result["predictions"]
    point_predictions_rounded = prepare_inventory_forecast(point_predictions)
    point_raw_mae = mean_absolute_error(point_predictions, test_targets)
    point_rounded_mae = mean_absolute_error(point_predictions_rounded, test_targets)

    quantile_predictions = quantile_result["predictions"]
    median_predictions = extract_quantile(quantile_predictions, 0.5)
    median_predictions_rounded = prepare_inventory_forecast(median_predictions)
    quantile_policy_results = evaluate_quantile_policies(
        quantile_predictions, test_targets, QUANTILES
    )

    quantile_pinball = pinball_loss_np(quantile_predictions, test_targets, QUANTILES)
    quantile_median_mae = mean_absolute_error(median_predictions, test_targets)
    quantile_median_mae_rounded = mean_absolute_error(median_predictions_rounded, test_targets)
    baseline_mae = mean_absolute_error(baseline_predictions, test_targets)

    baseline_inventory = inventory_simulation(baseline_predictions, test_targets)
    point_inventory = inventory_simulation(point_predictions_rounded, test_targets)

    coverage_by_quantile = quantile_coverages(quantile_predictions, test_targets, QUANTILES)
    central_interval_coverage = interval_coverage(quantile_predictions, test_targets)

    print("\nOppsummering av trening:")
    print_training_summary("Point TCN", point_result["training_result"])
    print_training_summary("Quantile TCN", quantile_result["training_result"])

    print("\nKontroll av output-shape:")
    print("Quantile test predictions shape:", quantile_predictions.shape)
    print("Forventet shape: (num_samples, 3, 7)")

    print("\nForecast-metrics på testsett:")
    print("Baseline MAE:", f"{baseline_mae:.4f}")
    print("Point TCN MAE (rå):", f"{point_raw_mae:.4f}")
    print("Point TCN MAE (avrundet):", f"{point_rounded_mae:.4f}")
    print("Quantile TCN pinball loss:", f"{quantile_pinball:.4f}")
    print("Quantile TCN median MAE (rå):", f"{quantile_median_mae:.4f}")
    print("Quantile TCN median MAE (avrundet):", f"{quantile_median_mae_rounded:.4f}")

    print("\nPolicy-MAE på testsett:")
    for quantile in QUANTILES:
        print(
            f"Quantile policy MAE (q={quantile:.1f} som order-up-to):",
            f"{quantile_policy_results[quantile]['mae']:.4f}",
        )

    print("\nCoverage på testsett:")
    for quantile, coverage in coverage_by_quantile.items():
        print(
            f"Empirisk coverage for q={quantile:.1f}: {coverage:.3f} "
            f"(mål: {quantile:.1f})"
        )
    print("Sentral 10-90% intervall coverage:", f"{central_interval_coverage:.3f}")

    print("\nFørste kvantilprediksjon for testsettet:")
    print("q0.1:", np.round(extract_quantile(quantile_predictions[:1], 0.1)[0], 3))
    print("q0.5:", np.round(extract_quantile(quantile_predictions[:1], 0.5)[0], 3))
    print("q0.9:", np.round(extract_quantile(quantile_predictions[:1], 0.9)[0], 3))
    print("Første faktisk target:", test_targets[0].astype(int))

    print("\nInventory metrics på testsett:")
    print(
        "Baseline total cost: {:.2f}, stockout rate: {:.3f}, fill rate: {:.3f}".format(
            *baseline_inventory
        )
    )
    print(
        "Point TCN total cost: {:.2f}, stockout rate: {:.3f}, fill rate: {:.3f}".format(
            *point_inventory
        )
    )

    print("\nQuantile policy-sammenligning:")
    for quantile in QUANTILES:
        label = {
            0.1: "aggressiv",
            0.5: "balansert",
            0.9: "konservativ",
        }.get(quantile, "policy")
        total_cost, stockout_rate, fill_rate = quantile_policy_results[quantile]["inventory"]
        print(
            "q={:.1f} ({}) total cost: {:.2f}, stockout rate: {:.3f}, fill rate: {:.3f}".format(
                quantile,
                label,
                total_cost,
                stockout_rate,
                fill_rate,
            )
        )


if __name__ == "__main__":
    main()
