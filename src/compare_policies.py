import csv
import html
from pathlib import Path

import torch

import preprocessing
from baselines import inventory_simulation, mean_absolute_error, moving_window_baseline
from train_quantile import (
    QUANTILES,
    evaluate_quantile_policies,
    extract_quantile,
    interval_coverage,
    pinball_loss_np,
    prepare_inventory_forecast,
    quantile_coverages,
    run_point_forecast,
    run_quantile_forecast,
)
from training_utils import build_dataloader, set_seed, to_tensor

RESULTS_DIR = Path("results/milestone_4")
FORECAST_CSV_PATH = RESULTS_DIR / "forecast_metrics.csv"
DECISION_CSV_PATH = RESULTS_DIR / "decision_metrics.csv"
MAIN_ANSWERS_PATH = RESULTS_DIR / "main_answers.txt"
FORECAST_FIGURE_PATH = RESULTS_DIR / "forecast_mae.svg"
TOTAL_COST_FIGURE_PATH = RESULTS_DIR / "decision_total_cost.svg"
SERVICE_FIGURE_PATH = RESULTS_DIR / "decision_service_levels.svg"
COVERAGE_FIGURE_PATH = RESULTS_DIR / "quantile_coverage.svg"
SEED = 42


def format_float(value, decimals=4):
    if value is None:
        return "-"
    return f"{value:.{decimals}f}"


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

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train = to_tensor(preprocessing.X_train, add_channel_dim=True)
    y_train = to_tensor(preprocessing.y_train)
    X_val = to_tensor(preprocessing.X_val, add_channel_dim=True)
    y_val = to_tensor(preprocessing.y_val)
    X_test = to_tensor(preprocessing.X_test, add_channel_dim=True)
    y_test = to_tensor(preprocessing.y_test)
    test_targets = preprocessing.y_test

    train_loader = build_dataloader(X_train, y_train, batch_size=32, shuffle=True)
    val_loader = build_dataloader(X_val, y_val, batch_size=32)
    test_loader = build_dataloader(X_test, y_test, batch_size=32)

    print("Kjører policy-sammenligning på samme split som Milepæl 3.")
    print("Device:", device)
    print("Seed:", SEED)
    print("Cost assumptions: holding_cost=1.0, stockout_cost=5.0")

    baseline_predictions = moving_window_baseline(
        preprocessing.X_test, forecast_horizon=preprocessing.TARGET_WINDOW
    )
    baseline_mae_raw = mean_absolute_error(baseline_predictions, test_targets)
    baseline_mae_rounded = baseline_mae_raw
    baseline_inventory = inventory_simulation(baseline_predictions, test_targets)

    print("\nTrener/kjører point forecast TCN...")
    point_result = run_point_forecast(train_loader, val_loader, test_loader, device)
    point_predictions = point_result["predictions"]
    point_predictions_rounded = prepare_inventory_forecast(point_predictions)
    point_mae_raw = mean_absolute_error(point_predictions, test_targets)
    point_mae_rounded = mean_absolute_error(point_predictions_rounded, test_targets)
    point_inventory = inventory_simulation(point_predictions_rounded, test_targets)

    print("\nTrener/kjører quantile forecast TCN...")
    quantile_result = run_quantile_forecast(train_loader, val_loader, test_loader, device)
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
        title="Forecast Evaluation",
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
        title="Decision Evaluation",
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
        "Milepæl 4 hovedfunn",
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
            "q=0.1 og q=0.5 er aggressive og gir svært lav servicegrad, mens q=0.9 er konservativ "
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
            "En mer konservativ policy øker bestillingsnivået, men under de nåværende cost assumptions "
            "ser reduksjonen i stockouts ut til å være mer verdifull enn ekstra beholdning."
        ),
        (
            "6. Hva sier dette om forskjellen mellom forecast quality og decision quality? "
            "Beste forecast-MAE gir ikke automatisk beste lagerbeslutning. Point TCN er bedre enn baseline "
            "på forecast-feil, men dårligere på decision metrics, mens quantile q=0.9 har mye bedre decision performance."
        ),
        "",
        (
            "Fra point forecast til beste quantile-policy endres total cost med "
            f"{total_cost_delta_vs_point:.2f}, stockout rate med {stockout_delta_vs_point:.3f} "
            f"og fill rate med {fill_rate_delta_vs_point:.3f}."
        ),
    ]

    print("\nMain Answers")
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
    print(main_answers[2])
    print(main_answers[3])
    print(main_answers[4])
    print(main_answers[5])
    print(main_answers[6])
    print(main_answers[7])
    print(main_answers[9])

    save_csv(FORECAST_CSV_PATH, forecast_rows)
    save_csv(DECISION_CSV_PATH, decision_rows)
    save_text(MAIN_ANSWERS_PATH, "\n".join(main_answers))

    forecast_mae_labels = ["baseline", "point_tcn", "quantile_median"]
    forecast_mae_values = [baseline_mae_raw, point_mae_raw, quantile_median_mae_raw]
    save_svg_bar_chart(
        FORECAST_FIGURE_PATH,
        title="Forecast MAE (Raw)",
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
        title="Decision Metric: Total Cost",
        labels=decision_labels,
        values=total_cost_values,
        y_label="Total cost",
        color="#F97316",
        value_decimals=2,
    )

    save_svg_grouped_bar_chart(
        SERVICE_FIGURE_PATH,
        title="Service Levels by Policy",
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
        title="Quantile Coverage vs Target",
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

    print("\nLagrer resultater i:")
    print(f"- {FORECAST_CSV_PATH}")
    print(f"- {DECISION_CSV_PATH}")
    print(f"- {MAIN_ANSWERS_PATH}")
    print(f"- {FORECAST_FIGURE_PATH}")
    print(f"- {TOTAL_COST_FIGURE_PATH}")
    print(f"- {SERVICE_FIGURE_PATH}")
    print(f"- {COVERAGE_FIGURE_PATH}")


if __name__ == "__main__":
    main()
