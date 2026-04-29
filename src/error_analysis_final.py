
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import preprocessing_5a
from baselines import moving_window_baseline
from models.tcn import QuantileTCN, SimpleTCN
from training_utils import to_tensor



POINT_CKPT = Path("data/processed/webapp_models/point_tcn_5a.pt")
QUANTILE_CKPT = Path("data/processed/webapp_models/quantile_tcn_5a.pt")
RESULTS_DIR = Path("results/error_analysis_final")
QUANTILES = (0.1, 0.5, 0.9)
T = 28
BATCH_SIZE = 256


def load_test_data():
    """
    Returns:
        X_np:        np.ndarray, shape (N, 28)
        y_np:        np.ndarray, shape (N, 7)
        series_ids:  np.ndarray of length N (one entry per window)
    """
    X_np = np.asarray(preprocessing_5a.X_test, dtype=np.float32)
    y_np = np.asarray(preprocessing_5a.y_test, dtype=np.float32)
    if X_np.ndim != 2 or X_np.shape[1] != T:
        raise ValueError(f"Expected X_test shape (N, {T}), got {X_np.shape}")

    series_ids_unique = np.asarray(preprocessing_5a.series_ids)
    n_series = len(series_ids_unique)
    n_windows = len(X_np)

    if n_windows % n_series == 0:
        windows_per_series = n_windows // n_series
        series_ids = np.repeat(series_ids_unique, windows_per_series)
    else:
        print(
            f"WARN: {n_windows} test windows do not divide evenly by {n_series} "
            f"series. Per-series stats will use synthetic group ids."
        )
        series_ids = np.array([f"window_{i:05d}" for i in range(n_windows)])

    return X_np, y_np, series_ids


def build_point_model(device: torch.device) -> torch.nn.Module:
    model = SimpleTCN(
        input_length=preprocessing_5a.INPUT_WINDOW,
        output_size=preprocessing_5a.TARGET_WINDOW,
        hidden_channels=32,
    ).to(device)
    if not POINT_CKPT.exists():
        raise FileNotFoundError(
            f"Point TCN checkpoint not found at {POINT_CKPT}. "
            "Run prepare_webapp_models.py (or your equivalent) first."
        )
    ckpt = torch.load(POINT_CKPT, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    
    model.eval()
    return model


def build_quantile_model(device: torch.device) -> torch.nn.Module:
    model = QuantileTCN(
        input_length=preprocessing_5a.INPUT_WINDOW,
        output_size=preprocessing_5a.TARGET_WINDOW,
        hidden_channels=32,
        num_quantiles=len(QUANTILES),
    ).to(device)
    if not QUANTILE_CKPT.exists():
        raise FileNotFoundError(
            f"Quantile TCN checkpoint not found at {QUANTILE_CKPT}. "
            "Run prepare_webapp_models.py (or your equivalent) first."
        )
    ckpt = torch.load(QUANTILE_CKPT, map_location=device)
    model.load_state_dict(ckpt["state_dict"])

    model.eval()
    return model


def predict_np(model: torch.nn.Module, X_np: np.ndarray,
               device: torch.device) -> np.ndarray:
    """Run model on a 2D numpy array (N, 28). Adds the channel dim internally."""
    out = []
    with torch.no_grad():
        for i in range(0, len(X_np), BATCH_SIZE):
            xb = to_tensor(X_np[i : i + BATCH_SIZE], add_channel_dim=True).to(device)
            out.append(model(xb).cpu().numpy())
    return np.concatenate(out, axis=0)


def run_error_analysis(X_np, y_np, series_ids, naive, point_pred, qmedian,
                       lines: list[str]) -> None:
    mae_naive = np.abs(naive - y_np).mean(axis=1)
    mae_point = np.abs(point_pred - y_np).mean(axis=1)
    mae_qmed = np.abs(qmedian - y_np).mean(axis=1)

    df = pd.DataFrame({
        "series_id": series_ids,
        "mean_demand": X_np.mean(axis=1),
        "mae_naive": mae_naive,
        "mae_point": mae_point,
        "mae_qmedian": mae_qmed,
    })

    per_series = (
        df.groupby("series_id", as_index=False)
        .agg(
            mae_naive=("mae_naive", "mean"),
            mae_point=("mae_point", "mean"),
            mae_qmedian=("mae_qmedian", "mean"),
            mean_demand=("mean_demand", "mean"),
        )
    )
    per_series.to_csv(RESULTS_DIR / "per_series.csv", index=False)

    wins_point = int((per_series["mae_point"] < per_series["mae_naive"]).sum())
    wins_qmed = int((per_series["mae_qmedian"] < per_series["mae_naive"]).sum())
    n = len(per_series)

    worst10 = per_series.nlargest(10, "mae_point")
    best10 = per_series.nsmallest(10, "mae_point")

    horizon_df = pd.DataFrame({
        "horizon_day": np.arange(1, 8),
        "mae_naive": np.abs(naive - y_np).mean(axis=0),
        "mae_point": np.abs(point_pred - y_np).mean(axis=0),
        "mae_qmedian": np.abs(qmedian - y_np).mean(axis=0),
    })
    horizon_df.to_csv(RESULTS_DIR / "per_horizon_day.csv", index=False)

    bucket_edges = per_series["mean_demand"].quantile([0, 1 / 3, 2 / 3, 1.0]).to_numpy(copy=True)
    bucket_edges[-1] += 1e-9
    per_series["demand_bucket"] = pd.cut(
        per_series["mean_demand"],
        bins=bucket_edges,
        labels=["low", "mid", "high"],
        include_lowest=True,
    )
    bucket_table = (
        per_series.groupby("demand_bucket", observed=True)
        [["mae_naive", "mae_point", "mae_qmedian"]]
        .mean()
        .round(4)
    )

    lines.append("Error analysis (5A test split)")
    lines.append("=" * 60)
    lines.append(f"Series evaluated:                 {n}")
    lines.append(f"Total windows:                    {len(X_np)}")
    lines.append("")
    lines.append("Mean MAE across all windows:")
    lines.append(f"  Naive baseline                  {mae_naive.mean():.4f}")
    lines.append(f"  Point TCN                       {mae_point.mean():.4f}")
    lines.append(f"  Quantile TCN (median)           {mae_qmed.mean():.4f}")
    lines.append("")
    lines.append("Win rate vs naive baseline (per series):")
    lines.append(f"  Point TCN beats baseline on     {wins_point}/{n}  ({wins_point / n:.1%})")
    lines.append(f"  Quantile median beats baseline  {wins_qmed}/{n}  ({wins_qmed / n:.1%})")
    lines.append("")
    lines.append("MAE per forecast horizon day:")
    lines.append(horizon_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    lines.append("")
    lines.append("MAE by mean-demand bucket (tertiles):")
    lines.append(bucket_table.to_string())
    lines.append("")
    lines.append("Worst 10 series for point TCN:")
    lines.append(
        worst10[["series_id", "mae_naive", "mae_point", "mae_qmedian", "mean_demand"]]
        .to_string(index=False, float_format=lambda v: f"{v:.4f}")
    )
    lines.append("")
    lines.append("Best 10 series for point TCN:")
    lines.append(
        best10[["series_id", "mae_naive", "mae_point", "mae_qmedian", "mean_demand"]]
        .to_string(index=False, float_format=lambda v: f"{v:.4f}")
    )
    lines.append("")

    # Figures
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.27
    x = np.arange(1, 8)
    ax.bar(x - width, horizon_df["mae_naive"], width, label="Baseline")
    ax.bar(x, horizon_df["mae_point"], width, label="Point TCN")
    ax.bar(x + width, horizon_df["mae_qmedian"], width, label="Quantile TCN (median)")
    ax.set_xlabel("Forecast horizon day")
    ax.set_ylabel("MAE")
    ax.set_title("MAE per forecast horizon day (5A test split)")
    ax.set_xticks(x)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig_mae_per_horizon.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(per_series["mean_demand"], per_series["mae_naive"],
               s=14, alpha=0.4, label="Baseline")
    ax.scatter(per_series["mean_demand"], per_series["mae_point"],
               s=14, alpha=0.6, label="Point TCN")
    ax.set_xlabel("Mean demand of series")
    ax.set_ylabel("Per-series MAE")
    ax.set_title("Per-series MAE vs. mean demand level")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig_mae_vs_demand.png", dpi=150)
    plt.close(fig)


def occlusion_curve(model, X, base_pred, is_quantile, device):
    if is_quantile:
        sens = np.zeros((3, T))
    else:
        sens = np.zeros(T)

    window_mean = X.mean(axis=1, keepdims=True)  # (N, 1)

    for t in range(T):
        Xo = X.copy()
        Xo[:, t] = window_mean[:, 0]
        occ = predict_np(model, Xo, device)
        if is_quantile:
            for q in range(3):
                sens[q, t] = np.abs(occ[:, q, :] - base_pred[:, q, :]).mean()
        else:
            sens[t] = np.abs(occ - base_pred).mean()

    return sens


def run_occlusion(point_model, quant_model, X_np, base_point, base_quant,
                  device, lines: list[str]) -> None:
    sens_point = occlusion_curve(point_model, X_np, base_point,
                                 is_quantile=False, device=device)
    sens_quant = occlusion_curve(quant_model, X_np, base_quant,
                                 is_quantile=True, device=device)
    np.save(RESULTS_DIR / "occlusion_point.npy", sens_point)
    np.save(RESULTS_DIR / "occlusion_quantile.npy", sens_quant)

    lines.append("Occlusion sensitivity")
    lines.append("=" * 60)
    lines.append("Each value is the mean absolute change in the forecast when a")
    lines.append("single day in the 28-day input window is replaced with the")
    lines.append("window's mean. Day 0 = most recent observed day.")
    lines.append("")

    lines.append("Point TCN - top 5 most influential input days:")
    order = np.argsort(sens_point)[::-1]
    for rank, idx in enumerate(order[:5], 1):
        days_back = T - 1 - idx
        lines.append(
            f"  Rank {rank}: day -{days_back:>2d}   sensitivity = {sens_point[idx]:.4f}"
        )
    lines.append("")

    lines.append("Quantile TCN - top 3 most influential input days per quantile:")
    for q_idx, q in enumerate(QUANTILES):
        lines.append(f"  q = {q}")
        order = np.argsort(sens_quant[q_idx])[::-1]
        for rank, idx in enumerate(order[:3], 1):
            days_back = T - 1 - idx
            lines.append(
                f"    Rank {rank}: day -{days_back:>2d}   sensitivity = {sens_quant[q_idx, idx]:.4f}"
            )
    lines.append("")

    recent_share_point = sens_point[-7:].sum() / sens_point.sum()
    lines.append(
        f"Share of total sensitivity from the most recent 7 days (point TCN): "
        f"{recent_share_point:.1%}"
    )
    for q_idx, q in enumerate(QUANTILES):
        recent_share_q = sens_quant[q_idx, -7:].sum() / sens_quant[q_idx].sum()
        lines.append(
            f"Share of total sensitivity from the most recent 7 days (q={q}): "
            f"{recent_share_q:.1%}"
        )
    lines.append("")

    # Figures
    fig, ax = plt.subplots(figsize=(8, 4))
    days_back = -np.arange(T - 1, -1, -1)
    ax.bar(days_back, sens_point, color="#3b6ea5")
    ax.set_xlabel("Day in input window (0 = most recent)")
    ax.set_ylabel("Mean absolute forecast change")
    ax.set_title("Occlusion sensitivity per input day - point TCN")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig_occlusion_point.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#7c3a89", "#888888", "#1f7a1f"]
    for q in range(3):
        ax.plot(days_back, sens_quant[q], marker="o", markersize=3,
                color=colors[q], label=f"q={QUANTILES[q]}")
    ax.set_xlabel("Day in input window (0 = most recent)")
    ax.set_ylabel("Mean absolute forecast change")
    ax.set_title("Occlusion sensitivity per input day - quantile TCN")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig_occlusion_quantile.png", dpi=150)
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading test data ...")
    X_np, y_np, series_ids = load_test_data()
    print(f"  N windows = {len(X_np)}, unique series = {len(np.unique(series_ids))}")

    print("Loading models ...")
    point_model = build_point_model(device)
    quant_model = build_quantile_model(device)

    print("Running base predictions ...")
    naive = moving_window_baseline(X_np, forecast_horizon=preprocessing_5a.TARGET_WINDOW)
    base_point = predict_np(point_model, X_np, device)
    base_quant = predict_np(quant_model, X_np, device)
    if base_quant.ndim != 3 or base_quant.shape[1] != 3:
        raise ValueError(f"Expected quantile pred shape (N, 3, 7), got {base_quant.shape}")
    qmedian = base_quant[:, 1, :]

    summary_lines: list[str] = []

    print("Part 1: error analysis ...")
    run_error_analysis(X_np, y_np, series_ids, naive, base_point, qmedian, summary_lines)

    print("Part 2: occlusion sensitivity ...")
    run_occlusion(point_model, quant_model, X_np, base_point, base_quant,
                  device, summary_lines)

    summary = "\n".join(summary_lines)
    print("\n" + summary)
    (RESULTS_DIR / "summary.txt").write_text(summary, encoding="utf-8")
    print(f"\nWrote outputs to {RESULTS_DIR.resolve()}")


if __name__ == "__main__":
    main()