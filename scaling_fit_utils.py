from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "loss curves" / "gpt_loss+lrs.pkl"
OUTPUT_ROOT = ROOT / "fit_outputs"

SCHEDULES = ["cosine", "wsd", "811"]
HUBER_DELTA = 1e-3
PRIMARY_EVAL_START = 2048


def parse_sw_mode() -> bool:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sw-mode", choices=["fit", "fixed"], default="fit")
    return parser.parse_args().sw_mode == "fit"


def method_name_with_sw(base_method: str, fit_sw: bool) -> str:
    suffix = "with_sw" if fit_sw else "no_sw"
    return f"{base_method}_{suffix}"


@dataclass
class CurveData:
    name: str
    full_steps: np.ndarray
    full_lr: np.ndarray
    observed_steps: np.ndarray
    observed_loss: np.ndarray
    observed_lr: np.ndarray
    observed_mask: np.ndarray
    full_loss: np.ndarray
    missing_steps: list[int]
    peak_lr: float
    lr_max_abs_error: float

    def fit_steps(self, start_step: int, stride: int) -> np.ndarray:
        keep = self.observed_steps >= start_step
        keep &= (self.observed_steps - start_step) % stride == 0
        steps = self.observed_steps[keep]
        if self.observed_steps[-1] not in steps:
            steps = np.append(steps, self.observed_steps[-1])
        return steps.astype(np.int64)

    def losses_at(self, steps: np.ndarray) -> np.ndarray:
        return self.full_loss[steps]


def schedule_key(name: str) -> str:
    return f"M:100M_gpt_D:20B_scheduler:{name}_rope"


def load_curves() -> tuple[dict[str, CurveData], list[dict[str, Any]]]:
    raw = pd.read_pickle(DATA_PATH)
    curves: dict[str, CurveData] = {}
    diagnostics: list[dict[str, Any]] = []

    for name in SCHEDULES:
        data = raw[schedule_key(name)].copy().sort_values("step")
        observed_steps = data["step"].to_numpy(dtype=np.int64)
        observed_loss = data["Metrics/loss"].to_numpy(dtype=np.float64)
        observed_lr = data["lr"].to_numpy(dtype=np.float64)

        if observed_steps.min() != 0:
            raise ValueError(f"{name} starts at step {observed_steps.min()}, but this code expects step 0.")
        if np.any(np.diff(observed_steps) <= 0):
            raise ValueError(f"{name} has repeated or non-monotone step values.")

        full_steps = np.arange(0, int(observed_steps.max()) + 1, dtype=np.int64)
        missing = sorted(set(full_steps.tolist()) - set(observed_steps.tolist()))
        if len(missing) > 1:
            raise ValueError(f"{name} has more than one missing step: {missing[:10]}")

        full_lr = np.interp(full_steps, observed_steps, observed_lr).astype(np.float64)
        lr_error = float(np.max(np.abs(full_lr[observed_steps] - observed_lr)))
        if lr_error > 1e-12:
            raise ValueError(f"{name} LR reconstruction changed observed LR values; max error={lr_error:g}")

        observed_mask = np.zeros(len(full_steps), dtype=bool)
        observed_mask[observed_steps] = True
        full_loss = np.full(len(full_steps), np.nan, dtype=np.float64)
        full_loss[observed_steps] = observed_loss
        peak_lr = float(np.max(full_lr))

        curves[name] = CurveData(
            name=name,
            full_steps=full_steps.copy(),
            full_lr=full_lr,
            observed_steps=observed_steps,
            observed_loss=observed_loss,
            observed_lr=observed_lr,
            observed_mask=observed_mask,
            full_loss=full_loss,
            missing_steps=missing,
            peak_lr=peak_lr,
            lr_max_abs_error=lr_error,
        )

        diagnostics.append(
            {
                "schedule": name,
                "rows": int(len(data)),
                "num_full_steps": int(len(full_steps)),
                "step_min": int(observed_steps.min()),
                "step_max": int(observed_steps.max()),
                "missing_steps": missing,
                "peak_lr_from_lr_column": peak_lr,
                "lr_reconstruction_method": "observed LR column plus linear interpolation for a single missing step",
                "lr_max_abs_error_at_observed_steps": lr_error,
                "lr_before_after_missing": _missing_lr_context(full_lr, missing),
            }
        )

    return curves, diagnostics


def _missing_lr_context(full_lr: np.ndarray, missing: list[int]) -> dict[str, Any] | None:
    if not missing:
        return None
    step = missing[0]
    lo = max(0, step - 2)
    hi = min(len(full_lr), step + 3)
    return {str(i): float(full_lr[i]) for i in range(lo, hi)}


def huber_np(residual: np.ndarray, delta: float = HUBER_DELTA) -> np.ndarray:
    abs_r = np.abs(residual)
    return np.where(abs_r <= delta, 0.5 * residual * residual, delta * (abs_r - 0.5 * delta))


def compute_metrics(
    curve: CurveData,
    prediction: np.ndarray,
    method: str,
    window: str,
    start_step: int,
) -> dict[str, Any]:
    mask = curve.observed_mask.copy()
    mask &= curve.full_steps >= start_step
    y = curve.full_loss[mask]
    pred = np.asarray(prediction, dtype=np.float64)[mask]
    valid = np.isfinite(y) & np.isfinite(pred) & (y > 0) & (pred > 0)
    y = y[valid]
    pred = pred[valid]

    residual = y - pred
    log_residual = np.log(y) - np.log(pred)
    ss_res = float(np.sum(residual * residual))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))

    return {
        "method": method,
        "schedule": curve.name,
        "window": window,
        "start_step": int(start_step),
        "n_points": int(len(y)),
        "huber_log_mean": float(np.mean(huber_np(log_residual))),
        "huber_log_sum": float(np.sum(huber_np(log_residual))),
        "log_rmse": float(np.sqrt(np.mean(log_residual * log_residual))),
        "log_mae": float(np.mean(np.abs(log_residual))),
        "rmse": float(np.sqrt(np.mean(residual * residual))),
        "mae": float(np.mean(np.abs(residual))),
        "mape": float(np.mean(np.abs(residual / y))),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "final_observed": float(y[-1]),
        "final_predicted": float(pred[-1]),
        "final_abs_error": float(abs(y[-1] - pred[-1])),
        "final_log_error": float(math.log(y[-1]) - math.log(pred[-1])),
    }


def compute_metric_rows(method: str, curves: dict[str, CurveData], predictions: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in SCHEDULES:
        rows.append(compute_metrics(curves[name], predictions[name], method, "all_observed", 0))
        rows.append(compute_metrics(curves[name], predictions[name], method, "post_2048", PRIMARY_EVAL_START))
    return rows


def smooth(values: np.ndarray, window: int = 101) -> np.ndarray:
    return (
        pd.Series(values, dtype="float64")
        .rolling(window=window, min_periods=1, center=True)
        .mean()
        .to_numpy(dtype=np.float64)
    )


def plot_three_schedules(
    method: str,
    curves: dict[str, CurveData],
    predictions: dict[str, np.ndarray],
    metrics_rows: list[dict[str, Any]],
    out_path: Path,
    smooth_window: int = 101,
) -> None:
    metric_lookup = {
        (row["schedule"], row["window"]): row
        for row in metrics_rows
    }
    colors = {"cosine": "#1f77b4", "wsd": "#2ca02c", "811": "#d62728"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)

    for ax, name in zip(axes, SCHEDULES):
        curve = curves[name]
        pred = predictions[name]
        obs_steps = curve.observed_steps
        obs_loss = curve.observed_loss
        ax.plot(obs_steps, smooth(obs_loss, smooth_window), color=colors[name], lw=1.8, label="observed (smoothed)")
        ax.plot(curve.full_steps, smooth(pred, smooth_window), color="black", lw=1.4, ls="--", label="prediction")
        if curve.missing_steps:
            ax.axvline(curve.missing_steps[0], color="#888888", lw=0.8, ls=":", alpha=0.8)
        m = metric_lookup[(name, "post_2048")]
        ax.set_title(f"{name} | RMSE={m['rmse']:.4f}, R2={m['r2']:.4f}")
        ax.set_xlabel("step")
        ax.set_ylim(2.5, 3.5)
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("loss")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle(method)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def save_predictions(method_dir: Path, curves: dict[str, CurveData], predictions: dict[str, np.ndarray]) -> None:
    method_dir.mkdir(parents=True, exist_ok=True)
    for name in SCHEDULES:
        curve = curves[name]
        frame = pd.DataFrame(
            {
                "step": curve.full_steps,
                "lr": curve.full_lr,
                "observed_loss": curve.full_loss,
                "observed": curve.observed_mask,
                "predicted_loss": predictions[name],
            }
        )
        frame.to_csv(method_dir / f"{name}_predictions.csv", index=False)


def write_method_outputs(
    method: str,
    params: dict[str, Any],
    curves: dict[str, CurveData],
    predictions: dict[str, np.ndarray],
    fit_info: dict[str, Any],
) -> list[dict[str, Any]]:
    method_dir = OUTPUT_ROOT / method
    method_dir.mkdir(parents=True, exist_ok=True)

    rows = compute_metric_rows(method, curves, predictions)
    pd.DataFrame(rows).to_csv(method_dir / "metrics.csv", index=False)
    save_json(method_dir / "params.json", params)
    save_json(method_dir / "fit_info.json", fit_info)
    save_predictions(method_dir, curves, predictions)
    plot_three_schedules(method, curves, predictions, rows, method_dir / "fit_curves.png")
    return rows


def write_data_diagnostics(diagnostics: list[dict[str, Any]]) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    save_json(OUTPUT_ROOT / "data_diagnostics.json", diagnostics)


class Timer:
    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        self.elapsed = 0.0
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.elapsed = time.perf_counter() - self.start
