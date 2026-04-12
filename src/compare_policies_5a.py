import csv
import html
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import preprocessing_5a
from baselines import inventory_simulation, mean_absolute_error, moving_window_baseline
from models.tcn import QuantileTCN, SimpleTCN
from training_utils import build_dataloader, predict, set_seed, to_tensor, train_model

RESULTS_DIR = Path("results/milestone_5a")
FORECAST_CSV_PATH = RESULTS_DIR / "forecast_metrics.csv"
DECISION_CSV_PATH = RESULTS_DIR / "decision_metrics.csv"
MAIN_ANSWERS_PATH = RESULTS_DIR / "main_answers.txt"
FORECAST_FIGURE_PATH = RESULTS_DIR / "forecast_mae.svg"
TOTAL_COST_FIGURE_PATH = RESULTS_DIR / "decision_total_cost.svg"
SERVICE_FIGURE_PATH = RESULTS_DIR / "decision_service_levels.svg"
COVERAGE_FIGURE_PATH = RESULTS_DIR / "quantile_coverage.svg"
TIMINGS_PATH = RESULTS_DIR / "timings.txt"

BATCH_SIZE = 32
EPOCHS = 60
POINT_LEARNING_RATE = 1e-3
QUANTILE_LEARNING_RATE = 5e-4
SEED = 42
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MIN_DELTA = 1e-4
QUANTILES = (0.1, 0.5, 0.9)
HOLDING_COST = 1.0
STOCKOUT_COST = 5.0


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
            "inventory": inventory_simulation(
                policy_forecast,
                targets,
                holding_cost=HOLDING_COST,
                stockout_cost=STOCKOUT_COST,
            ),
        }

    return policy_results


def run_point_forecast(train_loader, val_loader, test_loader, device):
    set_seed(SEED)
    model = SimpleTCN(
        input_length=preprocessing_5a.INPUT_WINDOW,
        output_size=preprocessing_5a.TARGET_WINDOW,
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
        input_length=preprocessing_5a.INPUT_WINDOW,
        output_size=preprocessing_5a.TARGET_WINDOW,
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


def format_float(value, decimals=4):
    if value is None:
        return "-"
    return f"{value:.{decimals}f}"


def format_duration(seconds):
    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:d}h {minutes:02d}m {secs:02d}s"
    if minutes > 0:
        return f"{minutes:d}m {secs:02d}s"
    return f"{secs:d}s"


def describe_training_result(name, training_result, elapsed_seconds):
    stop_text = (
        f"early stopped at epoch {training_result['stopped_epoch']}"
        if training_result["stopped_early"]
        else f"completed {training_result['stopped_epoch']} epochs"
    )
    return (
        f"{name}: {format_duration(elapsed_seconds)} | "
        f"best_epoch={training_result['best_epoch']} | "
        f"best_val_loss={training_result['best_val_loss']:.4f} | "
        f"{stop_text}"
    )


def print_table(title, columns, rows):
    widths = []
    for index, column in enumerate(columns):
        values = [str(row[index]) for row in rows]
        widths.append(max(len(column), *(len(value) for value in values)))

    print(f"\n{title}")
    header = " | ".join(column.ljust(widths[index]) for index, column in enumerate(columns))
    separator = "-+-".join("-" * width for width in widths)
    print(header)
    print(separator)

    for row in rows:
        print(" | ".join(str(value).ljust(widths[index]) for index, value in enumerate(row)))


def save_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_text(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def svg_text(x, y, text, font_size=16, weight="normal", anchor="start", fill="#111111"):
    return (
        f'<text x="{x}" y="{y}" font-size="{font_size}" font-family="Arial, sans-serif" '
        f'font-weight="{weight}" text-anchor="{anchor}" fill="{fill}">'
        f"{html.escape(str(text))}</text>"
    )


def save_svg_bar_chart(
    path,
    title,
    labels,
    values,
    y_label,
    color="#4472C4",
    y_max=None,
    value_decimals=3,
):
    width = 960
    height = 560
    margin_left = 90
    margin_right = 40
    margin_top = 80
    margin_bottom = 120
    chart_width = width - margin_left - margin_right
    chart_height = height - margin_top - margin_bottom

    max_value = max(values) if values else 1.0
    if y_max is None:
        y_max = max_value * 1.15 if max_value > 0 else 1.0
    y_max = max(y_max, 1e-6)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(width / 2, 35, title, font_size=24, weight="bold", anchor="middle"),
        svg_text(18, 60, y_label, font_size=14, fill="#555555"),
    ]

    for tick_index in range(6):
        tick_value = y_max * tick_index / 5
        y = margin_top + chart_height - (tick_value / y_max) * chart_height
        lines.append(
            f'<line x1="{margin_left}" y1="{y}" x2="{margin_left + chart_width}" y2="{y}" '
            'stroke="#DDDDDD" stroke-width="1"/>'
        )
        lines.append(svg_text(margin_left - 12, y + 5, f"{tick_value:.2f}", font_size=12, anchor="end", fill="#666666"))

    lines.append(
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + chart_height}" stroke="#444444" stroke-width="2"/>'
    )
    lines.append(
        f'<line x1="{margin_left}" y1="{margin_top + chart_height}" x2="{margin_left + chart_width}" y2="{margin_top + chart_height}" stroke="#444444" stroke-width="2"/>'
    )

    step = chart_width / max(len(labels), 1)
    bar_width = step * 0.6

    for index, (label, value) in enumerate(zip(labels, values)):
        x = margin_left + index * step + (step - bar_width) / 2
        bar_height = (value / y_max) * chart_height
        y = margin_top + chart_height - bar_height
        lines.append(
            f'<rect x="{x}" y="{y}" width="{bar_width}" height="{bar_height}" fill="{color}" rx="4" ry="4"/>'
        )
        lines.append(
            svg_text(
                x + bar_width / 2,
                y - 8,
                f"{value:.{value_decimals}f}",
                font_size=12,
                anchor="middle",
                fill="#333333",
            )
        )
        lines.append(
            svg_text(
                x + bar_width / 2,
                margin_top + chart_height + 24,
                label,
                font_size=12,
                anchor="middle",
                fill="#333333",
            )
        )

    lines.append("</svg>")
    save_text(path, "\n".join(lines))


def save_svg_grouped_bar_chart(
    path,
    title,
    labels,
    series,
    y_label,
    y_max=None,
    value_decimals=3,
):
    width = 1100
    height = 620
    margin_left = 90
    margin_right = 50
    margin_top = 100
    margin_bottom = 130
    chart_width = width - margin_left - margin_right
    chart_height = height - margin_top - margin_bottom

    max_value = max(max(group["values"]) for group in series) if series else 1.0
    if y_max is None:
        y_max = max_value * 1.15 if max_value > 0 else 1.0
    y_max = max(y_max, 1e-6)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(width / 2, 35, title, font_size=24, weight="bold", anchor="middle"),
        svg_text(18, 60, y_label, font_size=14, fill="#555555"),
    ]

    legend_x = width - margin_right - 220
    legend_y = 42
    for index, group in enumerate(series):
        x = legend_x + index * 110
        lines.append(
            f'<rect x="{x}" y="{legend_y - 12}" width="18" height="18" fill="{group["color"]}" rx="3" ry="3"/>'
        )
        lines.append(svg_text(x + 26, legend_y + 2, group["name"], font_size=13, fill="#333333"))

    for tick_index in range(6):
        tick_value = y_max * tick_index / 5
        y = margin_top + chart_height - (tick_value / y_max) * chart_height
        lines.append(
            f'<line x1="{margin_left}" y1="{y}" x2="{margin_left + chart_width}" y2="{y}" '
            'stroke="#DDDDDD" stroke-width="1"/>'
        )
        lines.append(svg_text(margin_left - 12, y + 5, f"{tick_value:.2f}", font_size=12, anchor="end", fill="#666666"))

    lines.append(
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + chart_height}" stroke="#444444" stroke-width="2"/>'
    )
    lines.append(
        f'<line x1="{margin_left}" y1="{margin_top + chart_height}" x2="{margin_left + chart_width}" y2="{margin_top + chart_height}" stroke="#444444" stroke-width="2"/>'
    )

    group_step = chart_width / max(len(labels), 1)
    grouped_width = group_step * 0.72
    bar_width = grouped_width / max(len(series), 1)

    for label_index, label in enumerate(labels):
        group_start_x = margin_left + label_index * group_step + (group_step - grouped_width) / 2

        for series_index, group in enumerate(series):
            value = group["values"][label_index]
            x = group_start_x + series_index * bar_width
            bar_height = (value / y_max) * chart_height
            y = margin_top + chart_height - bar_height
            lines.append(
                f'<rect x="{x}" y="{y}" width="{bar_width * 0.88}" height="{bar_height}" fill="{group["color"]}" rx="4" ry="4"/>'
            )
            lines.append(
                svg_text(
                    x + (bar_width * 0.88) / 2,
                    y - 8,
                    f"{value:.{value_decimals}f}",
                    font_size=11,
                    anchor="middle",
                    fill="#333333",
                )
            )

        lines.append(
            svg_text(
                group_start_x + grouped_width / 2,
                margin_top + chart_height + 24,
                label,
                font_size=12,
                anchor="middle",
                fill="#333333",
            )
        )

    lines.append("</svg>")
    save_text(path, "\n".join(lines))


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    total_start = time.perf_counter()
    run_started_at = time.strftime("%Y-%m-%d %H:%M:%S")

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train = to_tensor(preprocessing_5a.X_train, add_channel_dim=True)
    y_train = to_tensor(preprocessing_5a.y_train)
    X_val = to_tensor(preprocessing_5a.X_val, add_channel_dim=True)
    y_val = to_tensor(preprocessing_5a.y_val)
    X_test = to_tensor(preprocessing_5a.X_test, add_channel_dim=True)
    y_test = to_tensor(preprocessing_5a.y_test)
    test_targets = preprocessing_5a.y_test

    train_loader = build_dataloader(X_train, y_train, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = build_dataloader(X_val, y_val, batch_size=BATCH_SIZE)
    test_loader = build_dataloader(X_test, y_test, batch_size=BATCH_SIZE)

    print("Kjører 5A-policy-sammenligning på strengere datagrunnlag.")
    print("Run started at:", run_started_at)
    print("Device:", device)
    print("Seed:", SEED)
    print("Cost assumptions: holding_cost=1.0, stockout_cost=5.0")
    print("Antall serier:", len(preprocessing_5a.series_ids))
    print("Series matrix shape:", preprocessing_5a.series_matrix.shape)
    print("Train/val/test window shapes:")
    print("X_train:", preprocessing_5a.X_train.shape, "y_train:", preprocessing_5a.y_train.shape)
    print("X_val:", preprocessing_5a.X_val.shape, "y_val:", preprocessing_5a.y_val.shape)
    print("X_test:", preprocessing_5a.X_test.shape, "y_test:", preprocessing_5a.y_test.shape)

    baseline_start = time.perf_counter()
    baseline_predictions = moving_window_baseline(
        preprocessing_5a.X_test, forecast_horizon=preprocessing_5a.TARGET_WINDOW
    )
    baseline_mae_raw = mean_absolute_error(baseline_predictions, test_targets)
    baseline_mae_rounded = baseline_mae_raw
    baseline_inventory = inventory_simulation(
        baseline_predictions,
        test_targets,
        holding_cost=HOLDING_COST,
        stockout_cost=STOCKOUT_COST,
    )
    baseline_elapsed = time.perf_counter() - baseline_start
    print("\nBaseline evaluering ferdig på", format_duration(baseline_elapsed))

    print("\nTrener/kjører point forecast TCN...")
    point_start = time.perf_counter()
    point_result = run_point_forecast(train_loader, val_loader, test_loader, device)
    point_elapsed = time.perf_counter() - point_start
    point_predictions = point_result["predictions"]
    point_predictions_rounded = prepare_inventory_forecast(point_predictions)
    point_mae_raw = mean_absolute_error(point_predictions, test_targets)
    point_mae_rounded = mean_absolute_error(point_predictions_rounded, test_targets)
    point_inventory = inventory_simulation(
        point_predictions_rounded,
        test_targets,
        holding_cost=HOLDING_COST,
        stockout_cost=STOCKOUT_COST,
    )
    print(describe_training_result("Point TCN", point_result["training_result"], point_elapsed))

    print("\nTrener/kjører quantile forecast TCN...")
    quantile_start = time.perf_counter()
    quantile_result = run_quantile_forecast(train_loader, val_loader, test_loader, device)
    quantile_elapsed = time.perf_counter() - quantile_start
    quantile_predictions = quantile_result["predictions"]
    quantile_pinball_loss = pinball_loss_np(quantile_predictions, test_targets, QUANTILES)
    coverage_by_quantile = quantile_coverages(quantile_predictions, test_targets, QUANTILES)
    central_interval_coverage = interval_coverage(quantile_predictions, test_targets)
    quantile_policy_results = evaluate_quantile_policies(
        quantile_predictions, test_targets, QUANTILES
    )
    quantile_median_predictions = extract_quantile(quantile_predictions, 0.5)
    quantile_median_rounded = prepare_inventory_forecast(quantile_median_predictions)
    quantile_median_mae_raw = mean_absolute_error(quantile_median_predictions, test_targets)
    quantile_median_mae_rounded = mean_absolute_error(quantile_median_rounded, test_targets)
    print(
        describe_training_result(
            "Quantile TCN", quantile_result["training_result"], quantile_elapsed
        )
    )

    forecast_rows = [
        {
            "model": "baseline",
            "mae_raw": baseline_mae_raw,
            "mae_rounded": baseline_mae_rounded,
            "pinball_loss": None,
            "coverage_q0.1": None,
            "coverage_q0.5": None,
            "coverage_q0.9": None,
            "interval_10_90_coverage": None,
        },
        {
            "model": "point_tcn",
            "mae_raw": point_mae_raw,
            "mae_rounded": point_mae_rounded,
            "pinball_loss": None,
            "coverage_q0.1": None,
            "coverage_q0.5": None,
            "coverage_q0.9": None,
            "interval_10_90_coverage": None,
        },
        {
            "model": "quantile_tcn_median",
            "mae_raw": quantile_median_mae_raw,
            "mae_rounded": quantile_median_mae_rounded,
            "pinball_loss": quantile_pinball_loss,
            "coverage_q0.1": coverage_by_quantile[0.1],
            "coverage_q0.5": coverage_by_quantile[0.5],
            "coverage_q0.9": coverage_by_quantile[0.9],
            "interval_10_90_coverage": central_interval_coverage,
        },
    ]

    decision_rows = [
        {
            "policy": "baseline",
            "family": "baseline",
            "quantile": None,
            "policy_style": "reference",
            "total_cost": baseline_inventory[0],
            "stockout_rate": baseline_inventory[1],
            "fill_rate": baseline_inventory[2],
        },
        {
            "policy": "point_tcn",
            "family": "point",
            "quantile": None,
            "policy_style": "point forecast",
            "total_cost": point_inventory[0],
            "stockout_rate": point_inventory[1],
            "fill_rate": point_inventory[2],
        },
    ]

    quantile_labels = {
        0.1: "aggressiv",
        0.5: "balansert",
        0.9: "konservativ",
    }

    for quantile in QUANTILES:
        total_cost, stockout_rate, fill_rate = quantile_policy_results[quantile]["inventory"]
        decision_rows.append(
            {
                "policy": f"quantile_q{quantile:.1f}",
                "family": "quantile",
                "quantile": quantile,
                "policy_style": quantile_labels[quantile],
                "total_cost": total_cost,
                "stockout_rate": stockout_rate,
                "fill_rate": fill_rate,
            }
        )

    forecast_table_rows = [
        [
            row["model"],
            format_float(row["mae_raw"], 4),
            format_float(row["mae_rounded"], 4),
            format_float(row["pinball_loss"], 4),
            format_float(row["coverage_q0.1"], 3),
            format_float(row["coverage_q0.5"], 3),
            format_float(row["coverage_q0.9"], 3),
            format_float(row["interval_10_90_coverage"], 3),
        ]
        for row in forecast_rows
    ]

    decision_table_rows = [
        [
            row["policy"],
            row["policy_style"],
            format_float(row["total_cost"], 2),
            format_float(row["stockout_rate"], 3),
            format_float(row["fill_rate"], 3),
        ]
        for row in decision_rows
    ]

    print_table(
        title="Forecast Evaluation (5A)",
        columns=[
            "Model",
            "MAE Raw",
            "MAE Rounded",
            "Pinball Loss",
            "Cov q0.1",
            "Cov q0.5",
            "Cov q0.9",
            "10-90 Cov",
        ],
        rows=forecast_table_rows,
    )

    print_table(
        title="Decision Evaluation (5A)",
        columns=[
            "Policy",
            "Style",
            "Total Cost",
            "Stockout Rate",
            "Fill Rate",
        ],
        rows=decision_table_rows,
    )

    best_total_cost = min(decision_rows, key=lambda row: row["total_cost"])
    best_fill_rate = max(decision_rows, key=lambda row: row["fill_rate"])
    lowest_stockout = min(decision_rows, key=lambda row: row["stockout_rate"])
    best_quantile_policy = min(
        [row for row in decision_rows if row["family"] == "quantile"],
        key=lambda row: row["total_cost"],
    )
    point_policy = next(row for row in decision_rows if row["policy"] == "point_tcn")

    total_cost_delta_vs_point = best_quantile_policy["total_cost"] - point_policy["total_cost"]
    stockout_delta_vs_point = (
        best_quantile_policy["stockout_rate"] - point_policy["stockout_rate"]
    )
    fill_rate_delta_vs_point = best_quantile_policy["fill_rate"] - point_policy["fill_rate"]

    main_answers = [
        "Milepæl 5A hovedfunn",
        "",
        (
            "1. Slår point TCN baseline på forecast metrics? "
            f"Ja. Point TCN oppnår MAE raw {point_mae_raw:.4f} mot baseline {baseline_mae_raw:.4f}."
        ),
        (
            "2. Slår point TCN baseline på decision metrics? "
            f"Nei. Point TCN gir total cost {point_inventory[0]:.2f} mot baseline {baseline_inventory[0]:.2f}, "
            f"stockout rate {point_inventory[1]:.3f} mot {baseline_inventory[1]:.3f}, "
            f"og fill rate {point_inventory[2]:.3f} mot {baseline_inventory[2]:.3f}."
        ),
        (
            "3. Hvordan skiller q=0.1, q=0.5 og q=0.9 seg? "
            "q=0.1 og q=0.5 er mer aggressive og prioriterer lavere ordrevolum, mens q=0.9 er mer konservativ "
            "og reduserer stockouts kraftig."
        ),
        (
            "4. Hvilken kvantil fungerte best for inventory? "
            f"q=0.9, med total cost {best_quantile_policy['total_cost']:.2f}, "
            f"stockout rate {best_quantile_policy['stockout_rate']:.3f} og "
            f"fill rate {best_quantile_policy['fill_rate']:.3f}."
        ),
        (
            "5. Hva sier dette om trade-off mellom holding cost og stockout cost? "
            "Under disse cost assumptions ser reduksjonen i stockouts ut til å være mer verdifull "
            "enn den ekstra beholdningen som en konservativ policy krever."
        ),
        (
            "6. Hva sier dette om forskjellen mellom forecast quality og decision quality? "
            "Beste forecast-MAE gir ikke automatisk beste lagerbeslutning. Point TCN kan være bedre på forecast-feil "
            "uten å være bedre på decision metrics, mens quantile q=0.9 kan gi bedre beslutningsytelse."
        ),
        "",
        (
            "Fra point forecast til beste quantile-policy endres total cost med "
            f"{total_cost_delta_vs_point:.2f}, stockout rate med {stockout_delta_vs_point:.3f} "
            f"og fill rate med {fill_rate_delta_vs_point:.3f}."
        ),
    ]

    print("\nMain Answers (5A)")
    print(
        "Beste policy på total cost:",
        f"{best_total_cost['policy']} ({best_total_cost['total_cost']:.2f})",
    )
    print(
        "Beste policy på servicegrad (fill rate):",
        f"{best_fill_rate['policy']} ({best_fill_rate['fill_rate']:.3f})",
    )
    print(
        "Laveste stockout rate:",
        f"{lowest_stockout['policy']} ({lowest_stockout['stockout_rate']:.3f})",
    )

    total_elapsed = time.perf_counter() - total_start
    timing_lines = [
        "Milepæl 5A timing summary",
        f"run_started_at: {run_started_at}",
        f"device: {device}",
        f"baseline_eval: {format_duration(baseline_elapsed)}",
        describe_training_result("point_tcn", point_result["training_result"], point_elapsed),
        describe_training_result(
            "quantile_tcn", quantile_result["training_result"], quantile_elapsed
        ),
        f"total_runtime: {format_duration(total_elapsed)}",
    ]

    save_csv(FORECAST_CSV_PATH, forecast_rows)
    save_csv(DECISION_CSV_PATH, decision_rows)
    save_text(MAIN_ANSWERS_PATH, "\n".join(main_answers))
    save_text(TIMINGS_PATH, "\n".join(timing_lines))

    forecast_mae_labels = ["baseline", "point_tcn", "quantile_median"]
    forecast_mae_values = [baseline_mae_raw, point_mae_raw, quantile_median_mae_raw]
    save_svg_bar_chart(
        FORECAST_FIGURE_PATH,
        title="Forecast MAE (Raw) - 5A",
        labels=forecast_mae_labels,
        values=forecast_mae_values,
        y_label="MAE",
        color="#3B82F6",
        value_decimals=4,
    )

    decision_labels = [row["policy"] for row in decision_rows]
    total_cost_values = [row["total_cost"] for row in decision_rows]
    save_svg_bar_chart(
        TOTAL_COST_FIGURE_PATH,
        title="Decision Metric: Total Cost - 5A",
        labels=decision_labels,
        values=total_cost_values,
        y_label="Total cost",
        color="#F97316",
        value_decimals=2,
    )

    save_svg_grouped_bar_chart(
        SERVICE_FIGURE_PATH,
        title="Service Levels by Policy - 5A",
        labels=decision_labels,
        series=[
            {
                "name": "fill_rate",
                "values": [row["fill_rate"] for row in decision_rows],
                "color": "#10B981",
            },
            {
                "name": "stockout_rate",
                "values": [row["stockout_rate"] for row in decision_rows],
                "color": "#EF4444",
            },
        ],
        y_label="Rate",
        y_max=1.0,
        value_decimals=3,
    )

    save_svg_grouped_bar_chart(
        COVERAGE_FIGURE_PATH,
        title="Quantile Coverage vs Target - 5A",
        labels=[f"q={quantile:.1f}" for quantile in QUANTILES],
        series=[
            {
                "name": "target",
                "values": list(QUANTILES),
                "color": "#9CA3AF",
            },
            {
                "name": "empirical",
                "values": [coverage_by_quantile[quantile] for quantile in QUANTILES],
                "color": "#8B5CF6",
            },
        ],
        y_label="Coverage",
        y_max=1.0,
        value_decimals=3,
    )

    print("\nLagrer 5A-resultater i:")
    print(f"- {FORECAST_CSV_PATH}")
    print(f"- {DECISION_CSV_PATH}")
    print(f"- {MAIN_ANSWERS_PATH}")
    print(f"- {FORECAST_FIGURE_PATH}")
    print(f"- {TOTAL_COST_FIGURE_PATH}")
    print(f"- {SERVICE_FIGURE_PATH}")
    print(f"- {COVERAGE_FIGURE_PATH}")
    print(f"- {TIMINGS_PATH}")
    print("\nTiming Summary (5A)")
    for line in timing_lines:
        print(line)


if __name__ == "__main__":
    main()
