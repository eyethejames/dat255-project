import csv
import html
from pathlib import Path


RESULTS_ROOT = Path("results")
OUTPUT_DIR = RESULTS_ROOT / "report_figures"

MILESTONE_4_FORECAST = RESULTS_ROOT / "milestone_4" / "forecast_metrics.csv"
MILESTONE_4_DECISION = RESULTS_ROOT / "milestone_4" / "decision_metrics.csv"
MILESTONE_5A_FORECAST = RESULTS_ROOT / "milestone_5a" / "forecast_metrics.csv"
MILESTONE_5A_DECISION = RESULTS_ROOT / "milestone_5a" / "decision_metrics.csv"

FORECAST_OUTPUT = OUTPUT_DIR / "forecast_mae_m4_vs_5a.svg"
COVERAGE_OUTPUT = OUTPUT_DIR / "quantile_coverage_m4_vs_5a.svg"
SERVICE_OUTPUT = OUTPUT_DIR / "service_levels_m4_vs_5a.svg"
TOTAL_COST_M4_OUTPUT = OUTPUT_DIR / "total_cost_m4.svg"
TOTAL_COST_M5A_OUTPUT = OUTPUT_DIR / "total_cost_m5a.svg"

def read_csv_rows(path):
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def svg_text(x, y, text, font_size=16, weight="normal", anchor="start", fill="#111827"):
    return (
        f'<text x="{x}" y="{y}" font-size="{font_size}" font-family="Arial, sans-serif" '
        f'font-weight="{weight}" text-anchor="{anchor}" fill="{fill}">'
        f"{html.escape(str(text))}</text>"
    )


def value_or_zero(value):
    return 0.0 if value in (None, "", "-") else float(value)


def save_text(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def auto_y_max(values, padding=1.15):
    max_value = max(values) if values else 1.0
    if max_value <= 0:
        return 1.0
    return max_value * padding


def draw_centered_legend(lines, width, y, series, step=190):
    legend_width = max(len(series) - 1, 0) * step
    start_x = width / 2 - legend_width / 2 - 70

    for index, group in enumerate(series):
        x = start_x + index * step
        lines.append(
            f'<rect x="{x}" y="{y - 12}" width="18" height="18" fill="{group["color"]}" rx="3" ry="3"/>'
        )
        lines.append(svg_text(x + 26, y + 2, group["name"], font_size=13, fill="#374151"))


def save_grouped_bar_chart(
    path,
    title,
    subtitle,
    labels,
    series,
    y_label,
    y_max=None,
    value_decimals=3,
    show_legend=True,
):
    width = 1200
    height = 760
    margin_left = 90
    margin_right = 60
    margin_top = 115
    margin_bottom = 200
    chart_width = width - margin_left - margin_right
    chart_height = height - margin_top - margin_bottom

    max_value = max(max(group["values"]) for group in series) if series else 1.0
    if y_max is None:
        y_max = max_value * 1.15 if max_value > 0 else 1.0
    y_max = max(y_max, 1e-6)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(width / 2, 38, title, font_size=24, weight="bold", anchor="middle"),
        svg_text(width / 2, 65, subtitle, font_size=14, anchor="middle", fill="#4B5563"),
        svg_text(22, 92, y_label, font_size=14, fill="#4B5563"),
    ]

    for tick_index in range(6):
        tick_value = y_max * tick_index / 5
        y = margin_top + chart_height - (tick_value / y_max) * chart_height
        lines.append(
            f'<line x1="{margin_left}" y1="{y}" x2="{margin_left + chart_width}" y2="{y}" '
            'stroke="#E5E7EB" stroke-width="1"/>'
        )
        lines.append(
            svg_text(
                margin_left - 12,
                y + 5,
                f"{tick_value:.2f}",
                font_size=12,
                anchor="end",
                fill="#6B7280",
            )
        )

    lines.append(
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + chart_height}" stroke="#374151" stroke-width="2"/>'
    )
    lines.append(
        f'<line x1="{margin_left}" y1="{margin_top + chart_height}" x2="{margin_left + chart_width}" y2="{margin_top + chart_height}" stroke="#374151" stroke-width="2"/>'
    )

    group_step = chart_width / max(len(labels), 1)
    grouped_width = group_step * 0.74
    bar_width = grouped_width / max(len(series), 1)

    for label_index, label in enumerate(labels):
        group_start_x = margin_left + label_index * group_step + (group_step - grouped_width) / 2

        for series_index, group in enumerate(series):
            value = group["values"][label_index]
            x = group_start_x + series_index * bar_width
            bar_height = (value / y_max) * chart_height
            y = margin_top + chart_height - bar_height
            width_factor = bar_width * 0.82

            lines.append(
                f'<rect x="{x}" y="{y}" width="{width_factor}" height="{bar_height}" fill="{group["color"]}" rx="4" ry="4"/>'
            )
            lines.append(
                svg_text(
                    x + width_factor / 2,
                    y - 8,
                    f"{value:.{value_decimals}f}",
                    font_size=11,
                    anchor="middle",
                    fill="#1F2937",
                )
            )

        lines.append(
            svg_text(
                group_start_x + grouped_width / 2,
                margin_top + chart_height + 24,
                label,
                font_size=12,
                anchor="middle",
                fill="#374151",
            )
        )

    if show_legend:
        draw_centered_legend(lines, width, margin_top + chart_height + 82, series)
    lines.append("</svg>")
    save_text(path, "\n".join(lines))


def draw_panel(lines, x, y, width, height, title, labels, series, y_max, value_decimals):
    margin_left = 62
    margin_right = 28
    margin_top = 58
    margin_bottom = 72
    chart_x = x + margin_left
    chart_y = y + margin_top
    chart_width = width - margin_left - margin_right
    chart_height = height - margin_top - margin_bottom

    lines.append(
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" fill="none" stroke="#D1D5DB" stroke-width="1.5" rx="8" ry="8"/>'
    )
    lines.append(svg_text(x + width / 2, y + 30, title, font_size=18, weight="bold", anchor="middle"))

    for tick_index in range(6):
        tick_value = y_max * tick_index / 5
        tick_y = chart_y + chart_height - (tick_value / y_max) * chart_height
        lines.append(
            f'<line x1="{chart_x}" y1="{tick_y}" x2="{chart_x + chart_width}" y2="{tick_y}" stroke="#E5E7EB" stroke-width="1"/>'
        )
        lines.append(
            svg_text(
                chart_x - 10,
                tick_y + 5,
                f"{tick_value:.2f}",
                font_size=11,
                anchor="end",
                fill="#6B7280",
            )
        )

    lines.append(
        f'<line x1="{chart_x}" y1="{chart_y}" x2="{chart_x}" y2="{chart_y + chart_height}" stroke="#374151" stroke-width="2"/>'
    )
    lines.append(
        f'<line x1="{chart_x}" y1="{chart_y + chart_height}" x2="{chart_x + chart_width}" y2="{chart_y + chart_height}" stroke="#374151" stroke-width="2"/>'
    )

    group_step = chart_width / max(len(labels), 1)
    grouped_width = group_step * 0.74
    bar_width = grouped_width / max(len(series), 1)

    for label_index, label in enumerate(labels):
        group_start_x = chart_x + label_index * group_step + (group_step - grouped_width) / 2
        for series_index, group in enumerate(series):
            value = group["values"][label_index]
            bar_x = group_start_x + series_index * bar_width
            bar_height = (value / y_max) * chart_height
            bar_y = chart_y + chart_height - bar_height
            width_factor = bar_width * 0.82

            lines.append(
                f'<rect x="{bar_x}" y="{bar_y}" width="{width_factor}" height="{bar_height}" fill="{group["color"]}" rx="4" ry="4"/>'
            )
            lines.append(
                svg_text(
                    bar_x + width_factor / 2,
                    bar_y - 8,
                    f"{value:.{value_decimals}f}",
                    font_size=10,
                    anchor="middle",
                    fill="#1F2937",
                )
            )

        lines.append(
            svg_text(
                group_start_x + grouped_width / 2,
                chart_y + chart_height + 22,
                label,
                font_size=11,
                anchor="middle",
                fill="#374151",
            )
        )


def save_service_comparison_chart(path, title, subtitle, labels, fill_series, stockout_series):
    width = 1400
    height = 790
    panel_gap = 36
    outer_margin = 48
    panel_width = (width - 2 * outer_margin - panel_gap) / 2
    panel_height = 500
    panel_y = 120

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(width / 2, 38, title, font_size=24, weight="bold", anchor="middle"),
        svg_text(width / 2, 65, subtitle, font_size=14, anchor="middle", fill="#4B5563"),
    ]

    draw_panel(
        lines=lines,
        x=outer_margin,
        y=panel_y,
        width=panel_width,
        height=panel_height,
        title="Fill rate",
        labels=labels,
        series=fill_series,
        y_max=1.0,
        value_decimals=3,
    )
    draw_panel(
        lines=lines,
        x=outer_margin + panel_width + panel_gap,
        y=panel_y,
        width=panel_width,
        height=panel_height,
        title="Stockout rate",
        labels=labels,
        series=stockout_series,
        y_max=0.5,
        value_decimals=3,
    )

    draw_centered_legend(lines, width, panel_y + panel_height + 86, fill_series, step=210)
    lines.append("</svg>")
    save_text(path, "\n".join(lines))


def build_forecast_chart():
    forecast_m4 = {row["model"]: row for row in read_csv_rows(MILESTONE_4_FORECAST)}
    forecast_5a = {row["model"]: row for row in read_csv_rows(MILESTONE_5A_FORECAST)}

    labels = ["Baseline", "Point TCN", "Quantile TCN (median)"]
    save_grouped_bar_chart(
        path=FORECAST_OUTPUT,
        title="Forecast quality (MAE): M4 vs M5A",
        subtitle="Raw MAE values for baseline, point TCN and quantile TCN forecasts",
        labels=labels,
        series=[
            {
                "name": "Milestone 4",
                "values": [
                    value_or_zero(forecast_m4["baseline"]["mae_raw"]),
                    value_or_zero(forecast_m4["point_tcn"]["mae_raw"]),
                    value_or_zero(forecast_m4["quantile_tcn_median"]["mae_raw"]),
                ],
                "color": "#1D4ED8",
            },
            {
                "name": "Milestone 5A",
                "values": [
                    value_or_zero(forecast_5a["baseline"]["mae_raw"]),
                    value_or_zero(forecast_5a["point_tcn"]["mae_raw"]),
                    value_or_zero(forecast_5a["quantile_tcn_median"]["mae_raw"]),
                ],
                "color": "#F97316",
            },
        ],
        y_label="MAE (raw)",
        value_decimals=3,
    )


def build_coverage_chart():
    forecast_m4 = {row["model"]: row for row in read_csv_rows(MILESTONE_4_FORECAST)}
    forecast_5a = {row["model"]: row for row in read_csv_rows(MILESTONE_5A_FORECAST)}
    quantile_m4 = forecast_m4["quantile_tcn_median"]
    quantile_5a = forecast_5a["quantile_tcn_median"]

    labels = ["q=0.1", "q=0.5", "q=0.9", "10-90 interval"]
    save_grouped_bar_chart(
        path=COVERAGE_OUTPUT,
        title="Quantile calibration: Milestone 4 vs 5A",
        subtitle="Empirical coverage for quantile model compared to target coverage",
        labels=labels,
        series=[
            {
                "name": "Milestone 4",
                "values": [
                    value_or_zero(quantile_m4["coverage_q0.1"]),
                    value_or_zero(quantile_m4["coverage_q0.5"]),
                    value_or_zero(quantile_m4["coverage_q0.9"]),
                    value_or_zero(quantile_m4["interval_10_90_coverage"]),
                ],
                "color": "#7C3AED",
            },
            {
                "name": "Target coverage",
                "values": [0.1, 0.5, 0.9, 0.8],
                "color": "#9CA3AF",
            },
            {
                "name": "Milestone 5A",
                "values": [
                    value_or_zero(quantile_5a["coverage_q0.1"]),
                    value_or_zero(quantile_5a["coverage_q0.5"]),
                    value_or_zero(quantile_5a["coverage_q0.9"]),
                    value_or_zero(quantile_5a["interval_10_90_coverage"]),
                ],
                "color": "#0F766E",
            },
        ],
        y_label="Coverage",
        y_max=1.0,
        value_decimals=3,
    )


def build_service_chart():
    decision_m4 = {row["policy"]: row for row in read_csv_rows(MILESTONE_4_DECISION)}
    decision_5a = {row["policy"]: row for row in read_csv_rows(MILESTONE_5A_DECISION)}

    policy_order = [
        ("baseline", "Baseline"),
        ("point_tcn", "Point TCN"),
        ("quantile_q0.1", "Quantile q=0.1"),
        ("quantile_q0.5", "Quantile q=0.5"),
        ("quantile_q0.9", "Quantile q=0.9"),
    ]
    labels = [label for _, label in policy_order]

    fill_series = [
        {
            "name": "Milestone 4",
            "values": [value_or_zero(decision_m4[key]["fill_rate"]) for key, _ in policy_order],
            "color": "#1D4ED8",
        },
        {
            "name": "Milestone 5A",
            "values": [value_or_zero(decision_5a[key]["fill_rate"]) for key, _ in policy_order],
            "color": "#F97316",
        },
    ]
    stockout_series = [
        {
            "name": "Milestone 4",
            "values": [value_or_zero(decision_m4[key]["stockout_rate"]) for key, _ in policy_order],
            "color": "#1D4ED8",
        },
        {
            "name": "Milestone 5A",
            "values": [value_or_zero(decision_5a[key]["stockout_rate"]) for key, _ in policy_order],
            "color": "#F97316",
        },
    ]

    save_service_comparison_chart(
        path=SERVICE_OUTPUT,
        title="Service level: Milestone 4 vs 5A",
        subtitle="Fill rate and stockout rate comparison for same policy-set",
        labels=labels,
        fill_series=fill_series,
        stockout_series=stockout_series,
    )


def build_total_cost_charts():
    decision_m4 = {row["policy"]: row for row in read_csv_rows(MILESTONE_4_DECISION)}
    decision_5a = {row["policy"]: row for row in read_csv_rows(MILESTONE_5A_DECISION)}

    policy_order = [
        ("baseline", "Baseline"),
        ("point_tcn", "Point TCN"),
        ("quantile_q0.1", "Quantile q=0.1"),
        ("quantile_q0.5", "Quantile q=0.5"),
        ("quantile_q0.9", "Quantile q=0.9"),
    ]
    labels = [label for _, label in policy_order]
    m4_values = [value_or_zero(decision_m4[key]["total_cost"]) for key, _ in policy_order]
    m5a_values = [value_or_zero(decision_5a[key]["total_cost"]) for key, _ in policy_order]

    save_grouped_bar_chart(
        path=TOTAL_COST_M4_OUTPUT,
        title="Total cost by policy: Milestone 4",
        subtitle="Total cost values for baseline, point TCN and quantile policies",
        labels=labels,
        series=[
            {
                "name": "Milestone 4",
                "values": m4_values,
                "color": "#1D4ED8",
            }
        ],
        y_label="Total cost",
        y_max=auto_y_max(m4_values),
        value_decimals=0,
        show_legend=False,
    )

    save_grouped_bar_chart(
        path=TOTAL_COST_M5A_OUTPUT,
        title="Total cost by policy: Milestone 5A",
        subtitle="Total cost values for baseline, point TCN and quantile policies",
        labels=labels,
        series=[
            {
                "name": "Milestone 5A",
                "values": m5a_values,
                "color": "#F97316",
            }
        ],
        y_label="Total cost",
        y_max=auto_y_max(m5a_values),
        value_decimals=0,
        show_legend=False,
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    build_forecast_chart()
    build_coverage_chart()
    build_service_chart()
    build_total_cost_charts()

    print("Saved combined report figures in:")
    print(f"- {FORECAST_OUTPUT}")
    print(f"- {COVERAGE_OUTPUT}")
    print(f"- {SERVICE_OUTPUT}")
    print(f"- {TOTAL_COST_M4_OUTPUT}")
    print(f"- {TOTAL_COST_M5A_OUTPUT}")


if __name__ == "__main__":
    main()
