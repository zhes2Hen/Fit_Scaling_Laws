from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from scaling_fit_utils import (
    HUBER_DELTA,
    OUTPUT_ROOT,
    PRIMARY_EVAL_START,
    SCHEDULES,
    Timer,
    compute_metric_rows,
    huber_np,
    load_curves,
    method_name_with_sw,
    parse_sw_mode,
    save_json,
    write_data_diagnostics,
    write_method_outputs,
)


METHOD = "momentum_law"
DECAY_FACTOR = 0.999
FIT_START_STEP = PRIMARY_EVAL_START
FIT_STRIDE = 5


def compute_s1_s2(lr: np.ndarray, decay_factor: float = DECAY_FACTOR) -> tuple[np.ndarray, np.ndarray]:
    s1 = np.cumsum(lr, dtype=np.float64)
    momentum = np.zeros_like(lr, dtype=np.float64)
    for i in range(1, len(lr)):
        momentum[i] = decay_factor * momentum[i - 1] + (lr[i - 1] - lr[i])
    s2 = np.cumsum(momentum, dtype=np.float64)
    return s1, s2


class MomentumLaw:
    def __init__(self, schedule_terms: dict[str, tuple[np.ndarray, np.ndarray]], peak_lrs: dict[str, float]):
        self.schedule_terms = schedule_terms
        self.peak_lrs = peak_lrs

    def predict(self, schedule: str, params: np.ndarray) -> np.ndarray:
        l0, a, c, alpha = params[:4]
        s_w_prime = float(params[4]) if len(params) == 5 else 0.0
        s1, s2 = self.schedule_terms[schedule]
        shifted_s1 = np.maximum(s1 + self.peak_lrs[schedule] * s_w_prime, 1e-12)
        return l0 + a * np.power(shifted_s1, -alpha) - c * s2


def fit_model(fit_sw: bool) -> tuple[dict[str, float], dict[str, np.ndarray], dict[str, object]]:
    curves, diagnostics = load_curves()
    write_data_diagnostics(diagnostics)

    terms = {name: compute_s1_s2(curves[name].full_lr) for name in SCHEDULES}
    peak_lrs = {name: curves[name].peak_lr for name in SCHEDULES}
    model = MomentumLaw(terms, peak_lrs)

    train = curves["cosine"]
    fit_steps = train.fit_steps(FIT_START_STEP, FIT_STRIDE)
    fit_loss = train.losses_at(fit_steps)

    def objective(params: np.ndarray) -> float:
        pred = model.predict("cosine", params)[fit_steps]
        if np.any(~np.isfinite(pred)) or np.any(pred <= 0):
            return 1e12
        residual = np.log(fit_loss) - np.log(pred)
        return float(huber_np(residual, HUBER_DELTA).sum())

    min_loss = float(np.nanmin(train.full_loss[train.full_steps >= FIT_START_STEP]))
    l0_range = np.linspace(max(0.1, min_loss - 0.4), min_loss + 0.1, 3)
    a_range = [0.2, 1.0, 5.0, 20.0]
    c_range = [0.01, 0.1, 1.0, 10.0]
    alpha_range = [0.2, 0.5, 0.8, 1.2]
    sw_range = [0.0, 512.0, 2048.0, 8192.0] if fit_sw else [0.0]

    best_result = None
    best_value = float("inf")
    attempts = 0
    bounds = [(0.0, 10.0), (1e-12, 1e5), (0.0, 1e6), (0.0, 5.0)]
    if fit_sw:
        bounds.append((0.0, 1e6))

    for base_init in product(l0_range, a_range, c_range, alpha_range):
        for sw_init in sw_range:
            init = (*base_init, sw_init) if fit_sw else base_init
            attempts += 1
            result = minimize(
                objective,
                np.asarray(init, dtype=np.float64),
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 5000, "ftol": 1e-12, "gtol": 1e-9, "eps": 1e-8},
            )
            if result.fun < best_value:
                best_value = float(result.fun)
                best_result = result

    if best_result is None:
        raise RuntimeError("Momentum-law fitting failed to produce a result.")

    best_params = np.asarray(best_result.x, dtype=np.float64)
    if not fit_sw:
        best_params = np.asarray([*best_params, 0.0], dtype=np.float64)
    predictions = {name: model.predict(name, best_params) for name in SCHEDULES}
    params = {
        "L0": float(best_params[0]),
        "A": float(best_params[1]),
        "C": float(best_params[2]),
        "alpha": float(best_params[3]),
        "s_w_prime": float(best_params[4]),
        "S_W_cosine": float(curves["cosine"].peak_lr * best_params[4]),
        "decay_factor_lambda": DECAY_FACTOR,
        "fit_s_w_prime": bool(fit_sw),
    }
    fit_info = {
        "method": METHOD,
        "optimizer": "scipy.optimize.minimize L-BFGS-B with grid initializations",
        "fit_schedule": "cosine",
        "fit_start_step": FIT_START_STEP,
        "fit_stride": FIT_STRIDE,
        "fit_points": int(len(fit_steps)),
        "huber_delta": HUBER_DELTA,
        "best_training_huber_sum": best_value,
        "num_initializations": attempts,
        "scipy_success": bool(best_result.success),
        "scipy_message": str(best_result.message),
        "sw_mode": "fit" if fit_sw else "fixed",
        "peak_lrs": peak_lrs,
    }
    return params, predictions, fit_info


def main() -> None:
    fit_sw = parse_sw_mode()
    output_method = method_name_with_sw(METHOD, fit_sw)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with Timer() as timer:
        params, predictions, fit_info = fit_model(fit_sw)
    curves, _ = load_curves()
    fit_info["elapsed_seconds"] = timer.elapsed
    fit_info["base_method"] = METHOD
    fit_info["method"] = output_method
    rows = write_method_outputs(output_method, params, curves, predictions, fit_info)
    pd.DataFrame(compute_metric_rows(output_method, curves, predictions)).to_csv(OUTPUT_ROOT / output_method / "metrics.csv", index=False)
    save_json(OUTPUT_ROOT / output_method / "run_summary.json", {"params": params, "metrics": rows, "fit_info": fit_info})
    print(f"{output_method} done in {timer.elapsed:.2f}s")
    print(params)


if __name__ == "__main__":
    main()
