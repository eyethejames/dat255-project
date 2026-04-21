import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import preprocessing_5a
from models.tcn import QuantileTCN, SimpleTCN
from training_utils import build_dataloader, set_seed, to_tensor, train_model


ARTIFACT_DIR = Path("data/processed/webapp_models")
POINT_CHECKPOINT_PATH = ARTIFACT_DIR / "point_tcn_5a.pt"
QUANTILE_CHECKPOINT_PATH = ARTIFACT_DIR / "quantile_tcn_5a.pt"

INPUT_WINDOW = preprocessing_5a.INPUT_WINDOW
FORECAST_HORIZON = preprocessing_5a.TARGET_WINDOW
POINT_HIDDEN_CHANNELS = 32
QUANTILE_HIDDEN_CHANNELS = 32
POINT_LEARNING_RATE = 1e-3
QUANTILE_LEARNING_RATE = 5e-4
EPOCHS = 60
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MIN_DELTA = 1e-4
BATCH_SIZE = 32
SEED = 42
QUANTILES = (0.1, 0.5, 0.9)
HOLDING_COST = 1.0
STOCKOUT_COST = 5.0

_CACHED_MODELS = None


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


def log(message: str) -> None:
    print(message, file=sys.stderr)


def build_point_model():
    return SimpleTCN(
        input_length=INPUT_WINDOW,
        output_size=FORECAST_HORIZON,
        hidden_channels=POINT_HIDDEN_CHANNELS,
    )


def build_quantile_model():
    return QuantileTCN(
        input_length=INPUT_WINDOW,
        output_size=FORECAST_HORIZON,
        hidden_channels=QUANTILE_HIDDEN_CHANNELS,
        num_quantiles=len(QUANTILES),
    )


def prepare_training_loaders():
    X_train = to_tensor(preprocessing_5a.X_train, add_channel_dim=True)
    y_train = to_tensor(preprocessing_5a.y_train)
    X_val = to_tensor(preprocessing_5a.X_val, add_channel_dim=True)
    y_val = to_tensor(preprocessing_5a.y_val)

    train_loader = build_dataloader(X_train, y_train, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = build_dataloader(X_val, y_val, batch_size=BATCH_SIZE)
    return train_loader, val_loader


def checkpoint_payload(model, training_result, model_name):
    return {
        "model_name": model_name,
        "state_dict": model.state_dict(),
        "training_result": training_result,
        "input_window": INPUT_WINDOW,
        "forecast_horizon": FORECAST_HORIZON,
        "quantiles": list(QUANTILES),
    }


def save_checkpoint(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: Path, model_builder, device):
    checkpoint = torch.load(path, map_location=device)
    model = model_builder().to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint["training_result"]


def train_point_model(device, train_loader, val_loader):
    model = build_point_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=POINT_LEARNING_RATE)
    loss_fn = nn.L1Loss()

    with redirect_stdout(sys.stderr):
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

    save_checkpoint(
        POINT_CHECKPOINT_PATH,
        checkpoint_payload(model, training_result, "point_tcn_5a"),
    )
    model.eval()
    return model, training_result, True


def train_quantile_model(device, train_loader, val_loader):
    model = build_quantile_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=QUANTILE_LEARNING_RATE)
    loss_fn = PinballLoss(QUANTILES)

    with redirect_stdout(sys.stderr):
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

    save_checkpoint(
        QUANTILE_CHECKPOINT_PATH,
        checkpoint_payload(model, training_result, "quantile_tcn_5a"),
    )
    model.eval()
    return model, training_result, True


def ensure_models_ready():
    global _CACHED_MODELS

    if _CACHED_MODELS is not None:
        return _CACHED_MODELS

    set_seed(SEED)
    device = torch.device("cpu")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    train_loader = None
    val_loader = None

    if POINT_CHECKPOINT_PATH.exists():
        point_model, point_training_result = load_checkpoint(
            POINT_CHECKPOINT_PATH, build_point_model, device
        )
        point_trained_now = False
        log(f"Loaded point checkpoint from {POINT_CHECKPOINT_PATH}")
    else:
        log("Point checkpoint not found. Training point TCN for webapp inference...")
        train_loader, val_loader = prepare_training_loaders()
        point_model, point_training_result, point_trained_now = train_point_model(
            device, train_loader, val_loader
        )
        log(f"Saved point checkpoint to {POINT_CHECKPOINT_PATH}")

    if QUANTILE_CHECKPOINT_PATH.exists():
        quantile_model, quantile_training_result = load_checkpoint(
            QUANTILE_CHECKPOINT_PATH, build_quantile_model, device
        )
        quantile_trained_now = False
        log(f"Loaded quantile checkpoint from {QUANTILE_CHECKPOINT_PATH}")
    else:
        log("Quantile checkpoint not found. Training quantile TCN for webapp inference...")
        if train_loader is None or val_loader is None:
            train_loader, val_loader = prepare_training_loaders()
        quantile_model, quantile_training_result, quantile_trained_now = train_quantile_model(
            device, train_loader, val_loader
        )
        log(f"Saved quantile checkpoint to {QUANTILE_CHECKPOINT_PATH}")

    _CACHED_MODELS = {
        "device": device,
        "point_model": point_model,
        "quantile_model": quantile_model,
        "point_training_result": point_training_result,
        "quantile_training_result": quantile_training_result,
        "point_trained_now": point_trained_now,
        "quantile_trained_now": quantile_trained_now,
    }
    return _CACHED_MODELS


def normalize_series_id(raw_series_id: str) -> str:
    if raw_series_id.endswith("_validation"):
        return raw_series_id[: -len("_validation")]
    return raw_series_id


def series_lookup():
    normalized = [normalize_series_id(series_id) for series_id in preprocessing_5a.series_ids]
    return {series_id: index for index, series_id in enumerate(normalized)}


def load_series_values(series_id: str):
    lookup = series_lookup()
    if series_id not in lookup:
        raise KeyError(f"Unknown series_id: {series_id}")
    index = lookup[series_id]
    return preprocessing_5a.series_matrix[index].astype(np.float32)


def point_infer(model, history, device):
    input_tensor = torch.tensor(history, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        return model(input_tensor).cpu().numpy()[0]


def quantile_infer(model, history, device):
    input_tensor = torch.tensor(history, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        return model(input_tensor).cpu().numpy()[0]


def prepare_inventory_forecast(predictions):
    return np.clip(np.rint(np.asarray(predictions, dtype=np.float32)), 0, None)


def inventory_simulation(forecast, actual, holding_cost=HOLDING_COST, stockout_cost=STOCKOUT_COST):
    forecast = prepare_inventory_forecast(forecast)
    actual = np.asarray(actual, dtype=np.float32)

    total_cost = 0.0
    stockout_days = 0
    fulfilled_demand = 0.0
    total_demand = 0.0

    for predicted, demand in zip(forecast, actual):
        predicted_value = float(predicted)
        demand_value = float(demand)
        fulfilled = min(predicted_value, demand_value)
        leftover = max(0.0, predicted_value - demand_value)
        unmet = max(0.0, demand_value - predicted_value)

        total_cost += leftover * holding_cost + unmet * stockout_cost
        if unmet > 0:
            stockout_days += 1

        fulfilled_demand += fulfilled
        total_demand += demand_value

    return {
        "total_cost": total_cost,
        "stockout_rate": stockout_days / len(actual) if len(actual) > 0 else 0.0,
        "fill_rate": fulfilled_demand / total_demand if total_demand > 0 else 0.0,
    }


def baseline_infer(history):
    last_value = float(history[-1]) if len(history) > 0 else 0.0
    return np.full((FORECAST_HORIZON,), last_value, dtype=np.float32)


def make_serializable(values):
    return [float(value) for value in np.asarray(values, dtype=np.float32).tolist()]


def run_real_inference(series_id: str):
    values = load_series_values(series_id)
    if len(values) < INPUT_WINDOW + FORECAST_HORIZON:
        raise ValueError("Series is too short for real inference.")

    history = values[-(INPUT_WINDOW + FORECAST_HORIZON): -FORECAST_HORIZON]
    target = values[-FORECAST_HORIZON:]

    models = ensure_models_ready()
    device = models["device"]

    point_predictions = point_infer(models["point_model"], history, device)
    quantile_predictions = quantile_infer(models["quantile_model"], history, device)
    baseline_predictions = baseline_infer(history)

    q0_1 = quantile_predictions[0]
    q0_5 = quantile_predictions[1]
    q0_9 = quantile_predictions[2]

    policy_results = {
        "baseline": inventory_simulation(baseline_predictions, target),
        "point_tcn": inventory_simulation(point_predictions, target),
        "quantile_q0_9": inventory_simulation(q0_9, target),
    }

    recommended_policy = min(
        policy_results.items(),
        key=lambda item: item[1]["total_cost"],
    )[0]

    return {
        "series_id": series_id,
        "history": make_serializable(history),
        "target": make_serializable(target),
        "forecasts": {
            "point_tcn": make_serializable(point_predictions),
            "quantiles": {
                "q0_1": make_serializable(q0_1),
                "q0_5": make_serializable(q0_5),
                "q0_9": make_serializable(q0_9),
            },
        },
        "policy_results": policy_results,
        "recommended_policy": recommended_policy,
        "meta": {
            "input_window": INPUT_WINDOW,
            "forecast_horizon": FORECAST_HORIZON,
            "quantiles": list(QUANTILES),
            "inference_backend": "python_pytorch_tcn",
            "data_source": "M5 validation subset (CA_1 / FOODS / FOODS_1)",
            "point_model": {
                "checkpoint_path": str(POINT_CHECKPOINT_PATH),
                "best_epoch": models["point_training_result"]["best_epoch"],
                "best_val_loss": float(models["point_training_result"]["best_val_loss"]),
                "trained_now": models["point_trained_now"],
            },
            "quantile_model": {
                "checkpoint_path": str(QUANTILE_CHECKPOINT_PATH),
                "best_epoch": models["quantile_training_result"]["best_epoch"],
                "best_val_loss": float(models["quantile_training_result"]["best_val_loss"]),
                "trained_now": models["quantile_trained_now"],
            },
        },
    }


def warm_models():
    models = ensure_models_ready()
    return {
        "status": "ready",
        "point_checkpoint": str(POINT_CHECKPOINT_PATH),
        "quantile_checkpoint": str(QUANTILE_CHECKPOINT_PATH),
        "point_best_epoch": models["point_training_result"]["best_epoch"],
        "quantile_best_epoch": models["quantile_training_result"]["best_epoch"],
    }
