from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scaling_fit_utils as utils  # noqa: E402
from fit_multi_power_law import (  # noqa: E402
    FIT_START_STEP,
    FIT_STRIDE,
    predict_curve,
)
from staged_mpl import fit_base_only, fit_with_frozen_parameters  # noqa: E402


CONFIG_DIR = HERE / "configs"
OUTPUT_DIR = HERE / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the validation-guided three-stage MPL fitting protocol."
    )
    parser.add_argument("--stage1-fraction", type=float, default=0.30)
    parser.add_argument("--stage3-fraction", type=float, default=0.94)
    parser.add_argument("--fit-stride", type=int, default=FIT_STRIDE)
    parser.add_argument("--stage1-adam-steps", type=int, default=1800)
    parser.add_argument("--stage1-lbfgs-steps", type=int, default=100)
    parser.add_argument("--adam-steps", type=int, default=2500)
    parser.add_argument("--lbfgs-steps", type=int, default=120)
    parser.add_argument("--prefix", default="three_stage_mpl")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def rmse(curve, prediction: np.ndarray, start: int, end: int | None = None) -> float:
    mask = curve.observed_mask.copy()
    mask &= curve.full_steps >= start
    if end is not None:
        mask &= curve.full_steps <= end
    observed = curve.full_loss[mask]
    predicted = prediction[mask]
    valid = np.isfinite(observed) & np.isfinite(predicted)
    return float(np.sqrt(np.mean((observed[valid] - predicted[valid]) ** 2)))


def evaluation_rows(
    method: str,
    curves: dict,
    predictions: dict[str, np.ndarray],
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for schedule, curve in curves.items():
        boundary = int(math.floor(0.8 * int(curve.full_steps[-1])))
        rows.append(
            {
                "method": method,
                "schedule": schedule,
                "post_2048_rmse": rmse(
                    curve, predictions[schedule], FIT_START_STEP
                ),
                "stable_rmse": rmse(
                    curve, predictions[schedule], FIT_START_STEP, boundary - 1
                ),
                "decay_rmse": rmse(curve, predictions[schedule], boundary),
            }
        )
    return rows


def save_stage(
    method: str,
    params: dict,
    curves: dict,
    fit_info: dict,
) -> list[dict[str, float | str]]:
    predictions = {
        schedule: predict_curve(params, curves[schedule])
        for schedule in utils.SCHEDULES
    }
    utils.write_method_outputs(method, params, curves, predictions, fit_info)
    rows = evaluation_rows(method, curves, predictions)
    method_dir = OUTPUT_DIR / method
    pd.DataFrame(rows).to_csv(method_dir / "window_metrics.csv", index=False)
    return rows


def main() -> None:
    args = parse_args()
    if not 0.0 < args.stage1_fraction < 1.0:
        raise ValueError("--stage1-fraction must be in (0, 1).")
    if not 0.0 < args.stage3_fraction < 1.0:
        raise ValueError("--stage3-fraction must be in (0, 1).")

    # Keep generated artifacts inside this subfolder.
    utils.OUTPUT_ROOT = OUTPUT_DIR

    curves, diagnostics = utils.load_curves()
    cosine = curves["cosine"]
    baseline = load_json(CONFIG_DIR / "baseline_params.json")
    stage2_initial = load_json(CONFIG_DIR / "stage2_initial_params.json")

    all_steps = cosine.fit_steps(FIT_START_STEP, args.fit_stride)
    stage1_boundary = int(
        math.floor(args.stage1_fraction * int(cosine.full_steps[-1]))
    )
    stage1_steps = all_steps[all_steps <= stage1_boundary]
    base_params, stage1_details = fit_base_only(
        cosine,
        stage1_steps,
        baseline,
        adam_steps=args.stage1_adam_steps,
        lbfgs_steps=args.stage1_lbfgs_steps,
    )
    stage1_params = {
        **stage2_initial,
        **base_params,
        "fit_s_w_prime": True,
    }
    stage1_method = f"{args.prefix}_stage1_base"
    save_json(OUTPUT_DIR / stage1_method / "params.json", stage1_params)
    save_json(
        OUTPUT_DIR / stage1_method / "fit_info.json",
        {
            "stage": 1,
            "fit_schedule": "cosine",
            "fit_window": f"first {100 * args.stage1_fraction:.1f}% of progress",
            "model": "L0 + A * S1^(-alpha)",
            "purpose": "identify the base exponent without decay-kernel compensation",
            **stage1_details,
        },
    )

    stage2_params, stage2_details = fit_with_frozen_parameters(
        cosine,
        stage1_params,
        fixed_parameters={"alpha"},
        start_fraction=0.0,
        stride=args.fit_stride,
        adam_steps=args.adam_steps,
        lbfgs_steps=args.lbfgs_steps,
    )
    stage2_method = f"{args.prefix}_stage2_full_cosine"
    stage2_rows = save_stage(
        stage2_method,
        stage2_params,
        curves,
        {
            "stage": 2,
            "fit_schedule": "cosine",
            "fit_window": f"step {FIT_START_STEP} through the end",
            "fixed_parameters": ["alpha"],
            "purpose": "calibrate the global base level and initialize the decay kernel",
            **stage2_details,
        },
    )

    final_params, stage3_details = fit_with_frozen_parameters(
        cosine,
        stage2_params,
        fixed_parameters={"L0", "A", "alpha"},
        start_fraction=args.stage3_fraction,
        stride=args.fit_stride,
        adam_steps=args.adam_steps,
        lbfgs_steps=args.lbfgs_steps,
    )
    final_method = f"{args.prefix}_stage3_decay"
    final_rows = save_stage(
        final_method,
        final_params,
        curves,
        {
            "stage": 3,
            "fit_schedule": "cosine",
            "fit_window": (
                f"{100 * args.stage3_fraction:.1f}%-100% of training progress"
            ),
            "fixed_parameters": ["L0", "A", "alpha"],
            "purpose": "calibrate the late-decay response without moving the base trend",
            "protocol_note": (
                "The late-window boundary was selected using WSD as a validation "
                "schedule; 8-1-1 was retained as an additional transfer test."
            ),
            **stage3_details,
        },
    )

    summary = stage2_rows + final_rows
    pd.DataFrame(summary).to_csv(
        OUTPUT_DIR / f"{args.prefix}_summary.csv", index=False
    )
    save_json(OUTPUT_DIR / "data_diagnostics.json", diagnostics)

    print("Stage 1:", OUTPUT_DIR / stage1_method)
    print("Stage 2:", OUTPUT_DIR / stage2_method)
    print("Stage 3:", OUTPUT_DIR / final_method)


if __name__ == "__main__":
    main()
