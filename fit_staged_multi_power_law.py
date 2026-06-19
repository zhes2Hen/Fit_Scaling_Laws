import argparse
import math

import numpy as np
import pandas as pd


import scaling_fit_utils as utils
from fit_multi_power_law import (
    FIT_START_STEP,
    FIT_STRIDE,
    predict_curve,
)
from staged_multi_power_law import fit_base_only, fit_with_frozen_parameters


OUTPUT_DIR = utils.OUTPUT_ROOT
BASELINE_PARAMS = {
    "L0": 2.8193588256835938,
    "A": 1.4581011533737183,
    "alpha": 1.3618971109390259,
    "B": 449.1991882324219,
    "C": 0.11661767959594727,
    "beta": 0.23741869628429413,
    "gamma": 0.40688633918762207,
    "s_w_prime": 9.999998162868451e-09,
    "fit_s_w_prime": True,
}
STAGE2_INITIAL_PARAMS = {
    "L0": 2.723465919494629,
    "A": 1.274954915046692,
    "alpha": 0.9988600015640259,
    "B": 448.7814636230469,
    "C": 0.010856025852262974,
    "beta": 0.21318809688091278,
    "gamma": 0.5783498883247375,
    "s_w_prime": 9.999998162868451e-09,
    "fit_s_w_prime": True,
}


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
    parser.add_argument("--prefix", default="staged_multi_power_law")
    return parser.parse_args()


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

    curves, diagnostics = utils.load_curves()
    cosine = curves["cosine"]
    baseline = dict(BASELINE_PARAMS)
    stage2_initial = dict(STAGE2_INITIAL_PARAMS)

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
    utils.save_json(OUTPUT_DIR / stage1_method / "params.json", stage1_params)
    utils.save_json(
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
    utils.save_json(OUTPUT_DIR / "data_diagnostics.json", diagnostics)

    print("Stage 1:", OUTPUT_DIR / stage1_method)
    print("Stage 2:", OUTPUT_DIR / stage2_method)
    print("Stage 3:", OUTPUT_DIR / final_method)


if __name__ == "__main__":
    main()
