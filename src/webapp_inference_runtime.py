import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from models.tcn import QuantileTCN, SimpleTCN


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SERIES_PATH = PROJECT_ROOT / "webapp" / "data" / "ca_1_foods_1_validation_series.json"
ARTIFACT_DIR = PROJECT_ROOT / "data" / "processed" / "webapp_models"
POINT_CHECKPOINT_PATH = ARTIFACT_DIR / "point_tcn_5a.pt"
QUANTILE_CHECKPOINT_PATH = ARTIFACT_DIR / "quantile_tcn_5a.pt"

INPUT_WINDOW = 28
FORECAST_HORIZON = 7
POINT_HIDDEN_CHANNELS = 32
QUANTILE_HIDDEN_CHANNELS = 32
QUANTILES = (0.1, 0.5, 0.9)
HOLDING_COST = 1.0
STOCKOUT_COST = 5.0


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


def load_checkpoint(path: Path, model_builder):
    if not path.exists():
        raise FileNotFoundError(f"Missing model checkpoint: {path}")

    checkpoint = torch.load(path, map_location="cpu")
    model = model_builder()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint["training_result"]


def load_series_values(series_id: str):
    if not SERIES_PATH.exists():
        raise FileNotFoundError(f"Missing exported series data: {SERIES_PATH}")

    payload = json.loads(SERIES_PATH.read_text(encoding="utf-8"))
    for series in payload["series"]:
        if series["series_id"] == series_id:
            return np.asarray(series["values"], dtype=np.float32)

    raise KeyError(f"Unknown series_id: {series_id}")


def point_infer(model, history):
    input_tensor = torch.tensor(history, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        return model(input_tensor).cpu().numpy()[0]


def quantile_infer(model, history):
    input_tensor = torch.tensor(history, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        return model(input_tensor).cpu().numpy()[0]


def prepare_inventory_forecast(predictions):
    return np.clip(np.rint(np.asarray(predictions, dtype=np.float32)), 0, None)


def inventory_simulation(
    forecast,
    actual,
    holding_cost=HOLDING_COST,
    stockout_cost=STOCKOUT_COST,
):
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


def load_models():
    point_model, point_training_result = load_checkpoint(
        POINT_CHECKPOINT_PATH,
        build_point_model,
    )
    quantile_model, quantile_training_result = load_checkpoint(
        QUANTILE_CHECKPOINT_PATH,
        build_quantile_model,
    )
    return {
        "point_model": point_model,
        "quantile_model": quantile_model,
        "point_training_result": point_training_result,
        "quantile_training_result": quantile_training_result,
    }


def run_inference(series_id: str):
    values = load_series_values(series_id)
    if len(values) < INPUT_WINDOW + FORECAST_HORIZON:
        raise ValueError("Series is too short for real inference.")

    history = values[-(INPUT_WINDOW + FORECAST_HORIZON) : -FORECAST_HORIZON]
    target = values[-FORECAST_HORIZON:]

    models = load_models()

    point_predictions = point_infer(models["point_model"], history)
    quantile_predictions = quantile_infer(models["quantile_model"], history)
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
                "trained_now": False,
            },
            "quantile_model": {
                "checkpoint_path": str(QUANTILE_CHECKPOINT_PATH),
                "best_epoch": models["quantile_training_result"]["best_epoch"],
                "best_val_loss": float(
                    models["quantile_training_result"]["best_val_loss"]
                ),
                "trained_now": False,
            },
        },
    }


def warmup():
    models = load_models()
    return {
        "status": "ready",
        "series_path": str(SERIES_PATH),
        "point_checkpoint": str(POINT_CHECKPOINT_PATH),
        "quantile_checkpoint": str(QUANTILE_CHECKPOINT_PATH),
        "point_best_epoch": models["point_training_result"]["best_epoch"],
        "quantile_best_epoch": models["quantile_training_result"]["best_epoch"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--series-id", dest="series_id")
    parser.add_argument("--warmup-only", action="store_true")
    args = parser.parse_args()

    try:
        if args.warmup_only:
            payload = warmup()
        else:
            if not args.series_id:
                raise ValueError("--series-id is required unless --warmup-only is used.")
            payload = run_inference(args.series_id)
    except Exception as error:  # noqa: BLE001
        print(json.dumps({"error": str(error)}, ensure_ascii=True), file=sys.stdout)
        sys.exit(1)

    print(json.dumps(payload, ensure_ascii=True), file=sys.stdout)


if __name__ == "__main__":
    main()
