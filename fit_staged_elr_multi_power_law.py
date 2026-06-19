import argparse
import copy
import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch

from scaling_fit_utils import (
    HUBER_DELTA,
    OUTPUT_ROOT,
    PRIMARY_EVAL_START,
    SCHEDULES,
    Timer,
    compute_metric_rows,
    load_curves,
    save_json,
    write_method_outputs,
)
from fit_multi_power_law import inv_softplus, logit


METHOD = "staged_elr_multi_power_law"
FIT_START_STEP = PRIMARY_EVAL_START
FIT_STRIDE = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32

PARAMETERS = ("L0", "A", "alpha", "B", "C", "beta", "gamma", "C_u", "n0")
BASE_PARAMETERS = ("L0", "A", "alpha", "C_u", "n0")
DECAY_PARAMETERS = ("B", "C", "beta", "gamma")

STAGE1_END_FRACTIONS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60]
STAGE3_START_FRACTIONS = [0.90, 0.925, 0.935, 0.94, 0.945, 0.95, 0.955, 0.96]
ORACLE_DECAY_WEIGHTS = [1.0, 10.0, 20.0]
NORM_INITIALIZATION_GRID = [
    (1.0e-4, 1.0),
    (1.0, 10.0),
    (100.0, 1.0),
    (100.0, 10.0),
    (10000.0, 10.0),
]
STAGED_MPL_INITIAL_PARAMS = {
    "L0": 2.733398914337158,
    "A": 1.1800646781921387,
    "alpha": 0.946656346321106,
    "B": 448.7799072265625,
    "C": 0.003907037433236837,
    "beta": 0.07044423371553421,
    "gamma": 1.3417993783950806,
    "C_u": 100.0,
    "n0": 10.0,
}
REFERENCE_STAGED_MPL_METRICS = {
    "wsd_rmse": 0.040851,
    "811_rmse": 0.040692,
}

ADAM_LR = 2.0e-2
ADAM_MAX_STEPS = 2000
ADAM_MIN_STEPS = 400
ADAM_PATIENCE = 250
ADAM_MIN_DELTA = 5.0e-9
LBFGS_MAX_STEPS = 4
LBFGS_INNER_ITER = 30
LBFGS_PATIENCE = 3
LBFGS_MIN_DELTA = 1.0e-9
GRAD_TOL = 1.0e-7


def parse_float_list(text: str) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated number.")
    return values


def parse_norm_grid(text: str) -> list[tuple[float, float]]:
    pairs = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        left, right = item.split(":")
        pairs.append((float(left), float(right)))
    if not pairs:
        raise argparse.ArgumentTypeError("Expected entries like 100:10,1e-4:1.")
    return pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the staged ELR-MPL fitting protocol.")
    parser.add_argument("--fit-stride", type=int, default=FIT_STRIDE)
    parser.add_argument("--oracle-decay-weights", type=parse_float_list, default=ORACLE_DECAY_WEIGHTS)
    parser.add_argument("--stage1-end-fractions", type=parse_float_list, default=STAGE1_END_FRACTIONS)
    parser.add_argument("--stage3-start-fractions", type=parse_float_list, default=STAGE3_START_FRACTIONS)
    parser.add_argument(
        "--norm-initialization-grid",
        type=parse_norm_grid,
        default=NORM_INITIALIZATION_GRID,
        help="Comma-separated C_u:n0 pairs, for example 1e-4:1,100:10.",
    )
    return parser.parse_args()


def apply_runtime_args(args: argparse.Namespace) -> None:
    global FIT_STRIDE, ORACLE_DECAY_WEIGHTS, STAGE1_END_FRACTIONS, STAGE3_START_FRACTIONS, NORM_INITIALIZATION_GRID
    if args.fit_stride <= 0:
        raise ValueError("--fit-stride must be positive.")
    FIT_STRIDE = int(args.fit_stride)
    ORACLE_DECAY_WEIGHTS = list(args.oracle_decay_weights)
    STAGE1_END_FRACTIONS = list(args.stage1_end_fractions)
    STAGE3_START_FRACTIONS = list(args.stage3_start_fractions)
    NORM_INITIALIZATION_GRID = list(args.norm_initialization_grid)


@dataclass(frozen=True)
class FitSpec:
    schedule: str
    start_fraction: float = 0.0
    end_fraction: float = 1.0
    base_weight: float = 1.0
    wsd_decay_weight: float = 1.0
    fit_stride: int = FIT_STRIDE


@dataclass(frozen=True)
class FitConfig:
    name: str
    specs: list[FitSpec]
    fixed: set[str]
    base_only: bool = False
    parameter_regularization: float = 0.0
    adam_max_steps: int = ADAM_MAX_STEPS
    lbfgs_max_steps: int = LBFGS_MAX_STEPS


def tag(value: float) -> str:
    return str(value).replace(".", "p")


def softplus_value(x: float) -> float:
    return math.log(math.expm1(max(float(x), 1.0e-8))) if x <= 20 else float(x)


def raw_from_value(name: str, value: float) -> float:
    value = float(value)
    if name == "L0":
        return logit((value - 1.0) / 4.0)
    if name in {"A", "B", "C", "C_u", "n0"}:
        return softplus_value(value)
    if name in {"alpha", "beta", "gamma"}:
        return logit(value / 2.0)
    raise KeyError(name)


def value_from_raw(name: str, raw: torch.Tensor) -> torch.Tensor:
    if name == "L0":
        return 1.0 + 4.0 * torch.sigmoid(raw)
    if name in {"A", "B", "C", "C_u", "n0"}:
        return torch.nn.functional.softplus(raw) + 1.0e-10
    if name in {"alpha", "beta", "gamma"}:
        return 2.0 * torch.sigmoid(raw) + 1.0e-8
    raise KeyError(name)


def grad_max_norm(model: torch.nn.Module) -> float:
    result = 0.0
    for param in model.parameters():
        if param.grad is not None:
            result = max(result, float(param.grad.detach().abs().max().cpu()))
    return result


def log_huber(observed: torch.Tensor, predicted: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    residual = torch.log(observed) - torch.log(predicted.clamp_min(1.0e-8))
    absolute = residual.abs()
    huber = torch.where(
        absolute <= HUBER_DELTA,
        0.5 * residual.square(),
        HUBER_DELTA * (absolute - 0.5 * HUBER_DELTA),
    )
    return (weights * huber).sum() / weights.sum().clamp_min(1.0e-12)


class ELRMPLModel(torch.nn.Module):
    def __init__(self, initial: dict[str, float], fixed: set[str]):
        super().__init__()
        self.fixed = set(fixed)
        self.trainable = [name for name in PARAMETERS if name not in self.fixed]

        for name in self.fixed:
            self.register_buffer(
                f"fixed_{name}",
                torch.tensor(float(initial[name]), dtype=DTYPE, device=DEVICE),
            )

        raw_values = [raw_from_value(name, initial[name]) for name in self.trainable]
        raw = torch.tensor(raw_values, dtype=DTYPE, device=DEVICE)
        self.raw = torch.nn.Parameter(raw.clone(), requires_grad=bool(self.trainable))
        self.register_buffer("raw_reference", raw.clone())

    def value(self, name: str) -> torch.Tensor:
        if name in self.fixed:
            return getattr(self, f"fixed_{name}")
        return value_from_raw(name, self.raw[self.trainable.index(name)])

    def unpack(self) -> tuple[torch.Tensor, ...]:
        return tuple(self.value(name) for name in PARAMETERS)

    def params_dict(self) -> dict[str, float | bool | str]:
        values = self.unpack()
        result = {name: float(value.detach().cpu()) for name, value in zip(PARAMETERS, values)}
        result["lambda_wd"] = 0.0
        result["intrinsic_time"] = "sum eta_k / sqrt(n_k)"
        result["norm_recurrence"] = "n_{k+1}=n_k+eta_k^2*C_u"
        return result


class ELRMPLCache:
    def __init__(self, curve, steps: np.ndarray, weights: np.ndarray):
        max_step = int(np.max(steps))
        active_steps = np.arange(1, max_step + 1, dtype=np.int64)

        self.curve_name = curve.name
        self.steps = torch.tensor(steps, dtype=torch.long, device=DEVICE)
        self.loss = torch.tensor(curve.losses_at(steps), dtype=DTYPE, device=DEVICE)
        self.weights = torch.tensor(weights.astype(np.float32), dtype=DTYPE, device=DEVICE)
        self.lr = torch.tensor(curve.full_lr[: max_step + 1].astype(np.float32), dtype=DTYPE, device=DEVICE)
        self.active_steps = torch.tensor(active_steps, dtype=torch.long, device=DEVICE)
        self.step_min = int(steps.min())
        self.step_max = int(steps.max())
        self.fit_points = int(len(steps))

    def elr_quantities(self, C_u: torch.Tensor, n0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        increments = self.lr.square() * C_u
        sum_before = torch.cat(
            [
                torch.zeros(1, dtype=DTYPE, device=DEVICE),
                torch.cumsum(increments, dim=0)[:-1],
            ]
        )
        norm_square = (n0 + sum_before).clamp_min(1.0e-12)
        elr = self.lr / torch.sqrt(norm_square)
        s1 = torch.cumsum(elr, dim=0)
        return norm_square, elr, s1


def select_steps(curve, start_fraction: float, end_fraction: float, stride: int) -> np.ndarray:
    if not 0.0 <= start_fraction < 1.0:
        raise ValueError("start_fraction must be in [0, 1).")
    if not 0.0 < end_fraction <= 1.0:
        raise ValueError("end_fraction must be in (0, 1].")
    if start_fraction >= end_fraction:
        raise ValueError("start_fraction must be smaller than end_fraction.")
    last_step = int(curve.full_steps[-1])
    start = max(FIT_START_STEP, int(math.floor(start_fraction * last_step)))
    end = int(math.floor(end_fraction * last_step))
    steps = curve.observed_steps[(curve.observed_steps >= start) & (curve.observed_steps <= end)]
    steps = steps[(steps - start) % stride == 0]
    if len(steps) == 0 or steps[-1] != curve.observed_steps[curve.observed_steps <= end][-1]:
        steps = np.append(steps, curve.observed_steps[curve.observed_steps <= end][-1])
    return np.unique(steps.astype(np.int64))


def detect_wsd_decay_start(curve) -> int:
    lr = curve.full_lr
    drops = np.where(lr[1:] < lr[:-1] - 1.0e-15)[0] + 1
    return int(drops[0]) if len(drops) else int(curve.full_steps[-1])


def weights_for_steps(curve, steps: np.ndarray, spec: FitSpec) -> np.ndarray:
    weights = np.full(len(steps), float(spec.base_weight), dtype=np.float32)
    if curve.name == "wsd" and spec.wsd_decay_weight != spec.base_weight:
        decay_start = detect_wsd_decay_start(curve)
        weights[steps >= decay_start] = float(spec.wsd_decay_weight)
    return weights


def make_caches(curves: dict[str, Any], specs: list[FitSpec]) -> list[ELRMPLCache]:
    caches = []
    for spec in specs:
        curve = curves[spec.schedule]
        steps = select_steps(curve, spec.start_fraction, spec.end_fraction, spec.fit_stride)
        weights = weights_for_steps(curve, steps, spec)
        caches.append(ELRMPLCache(curve, steps, weights))
    return caches


def predict_from_cache(model: ELRMPLModel, cache: ELRMPLCache, base_only: bool = False) -> torch.Tensor:
    L0, A, alpha, B, C, beta, gamma, C_u, n0 = model.unpack()
    _, elr, s1 = cache.elr_quantities(C_u, n0)
    target_s1 = s1[cache.steps]
    base = L0 + A * target_s1.clamp_min(1.0e-12).pow(-alpha)
    if base_only:
        return base

    active = cache.active_steps
    drop = torch.relu(elr[active - 1] - elr[active])
    active_elr = elr[active].clamp_min(1.0e-12)
    active_before = s1[active - 1]
    span = (target_s1[:, None] - active_before[None, :]).clamp_min(0.0)
    mask = active[None, :] <= cache.steps[:, None]
    x = C * span * active_elr[None, :].pow(-gamma)
    kernel = 1.0 - (1.0 + x.clamp_min(0.0)).pow(-beta)
    loss_drop = (drop[None, :] * mask.to(DTYPE) * kernel).sum(dim=1)
    return base - B * loss_drop


def training_loss(model: ELRMPLModel, caches: list[ELRMPLCache], config: FitConfig) -> torch.Tensor:
    pieces = []
    for cache in caches:
        pred = predict_from_cache(model, cache, config.base_only)
        pieces.append(log_huber(cache.loss, pred, cache.weights))
        pieces.append(torch.relu(1.0e-6 - pred).square().mean() * 1.0e6)
    total = torch.stack(pieces).mean()
    if config.parameter_regularization > 0:
        total = total + float(config.parameter_regularization) * (model.raw - model.raw_reference).square().mean()
    return total


def fit_stage(
    curves: dict[str, Any],
    initial: dict[str, float],
    config: FitConfig,
) -> tuple[dict[str, float | bool | str], dict[str, Any]]:
    caches = make_caches(curves, config.specs)
    model = ELRMPLModel(initial, config.fixed).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=ADAM_LR)

    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    steps_since_improvement = 0
    adam_history = []
    adam_stop = "adam_max_steps"

    for step in range(config.adam_max_steps):
        optimizer.zero_grad(set_to_none=True)
        value = training_loss(model, caches, config)
        value.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        max_grad = grad_max_norm(model)
        optimizer.step()

        scalar = float(value.detach().cpu())
        if step % 50 == 0 or step == config.adam_max_steps - 1:
            adam_history.append({"step": step, "loss": scalar, "grad_max": max_grad})
        if scalar < best_loss - ADAM_MIN_DELTA:
            best_loss = scalar
            best_state = copy.deepcopy(model.state_dict())
            steps_since_improvement = 0
        else:
            steps_since_improvement += 1
        if step % 400 == 0:
            print(f"{config.name} Adam {step:4d}: {scalar:.6e}, best={best_loss:.6e}, grad={max_grad:.3e}")
        if step >= ADAM_MIN_STEPS and steps_since_improvement >= ADAM_PATIENCE:
            adam_stop = "adam_patience"
            break
        if step >= ADAM_MIN_STEPS and max_grad < GRAD_TOL:
            adam_stop = "adam_grad_tol"
            break

    model.load_state_dict(best_state)
    lbfgs = torch.optim.LBFGS(
        model.parameters(),
        lr=0.7,
        max_iter=LBFGS_INNER_ITER,
        line_search_fn="strong_wolfe",
    )
    lbfgs_history = []
    lbfgs_no_improve = 0
    lbfgs_stop = "lbfgs_max_steps"

    for outer in range(config.lbfgs_max_steps):
        def closure() -> torch.Tensor:
            lbfgs.zero_grad(set_to_none=True)
            total = training_loss(model, caches, config)
            total.backward()
            return total

        lbfgs.step(closure)
        lbfgs.zero_grad(set_to_none=True)
        current = training_loss(model, caches, config)
        current.backward()
        max_grad = grad_max_norm(model)
        scalar = float(current.detach().cpu())
        lbfgs_history.append({"step": outer, "loss": scalar, "grad_max": max_grad})
        print(f"{config.name} LBFGS {outer + 1}: {scalar:.6e}, grad={max_grad:.3e}")

        if scalar < best_loss - LBFGS_MIN_DELTA:
            best_loss = scalar
            best_state = copy.deepcopy(model.state_dict())
            lbfgs_no_improve = 0
        else:
            lbfgs_no_improve += 1
        if max_grad < GRAD_TOL:
            lbfgs_stop = "lbfgs_grad_tol"
            break
        if lbfgs_no_improve >= LBFGS_PATIENCE:
            lbfgs_stop = "lbfgs_patience"
            break

    model.load_state_dict(best_state)
    info = {
        "stage_name": config.name,
        "base_only": config.base_only,
        "fixed_parameters": sorted(config.fixed),
        "trainable_parameters": [name for name in PARAMETERS if name not in config.fixed],
        "fit_specs": [
            {
                "schedule": spec.schedule,
                "start_fraction": spec.start_fraction,
                "end_fraction": spec.end_fraction,
                "fit_stride": spec.fit_stride,
                "base_weight": spec.base_weight,
                "wsd_decay_weight": spec.wsd_decay_weight,
            }
            for spec in config.specs
        ],
        "fit_points": int(sum(cache.fit_points for cache in caches)),
        "fit_step_min": int(min(cache.step_min for cache in caches)),
        "fit_step_max": int(max(cache.step_max for cache in caches)),
        "optimizer": "Adam with early stopping followed by repeated LBFGS with early stopping",
        "adam_max_steps": config.adam_max_steps,
        "adam_min_steps": ADAM_MIN_STEPS,
        "adam_patience": ADAM_PATIENCE,
        "adam_min_delta": ADAM_MIN_DELTA,
        "adam_stop_reason": adam_stop,
        "lbfgs_max_steps": config.lbfgs_max_steps,
        "lbfgs_inner_iter": LBFGS_INNER_ITER,
        "lbfgs_patience": LBFGS_PATIENCE,
        "lbfgs_min_delta": LBFGS_MIN_DELTA,
        "lbfgs_stop_reason": lbfgs_stop,
        "best_training_loss": best_loss,
        "adam_history": adam_history,
        "lbfgs_history": lbfgs_history,
    }
    return model.params_dict(), info


def model_from_params(params: dict[str, float]) -> ELRMPLModel:
    initial = {name: float(params[name]) for name in PARAMETERS}
    return ELRMPLModel(initial, fixed=set(PARAMETERS)).to(DEVICE)


@torch.no_grad()
def predict_curve(params: dict[str, float], curve) -> np.ndarray:
    model = model_from_params(params)
    full_steps = curve.full_steps
    pieces = []
    for start in range(0, len(full_steps), 256):
        steps = full_steps[start : start + 256]
        weights = np.ones(len(steps), dtype=np.float32)
        cache = ELRMPLCache(curve, steps, weights)
        pieces.append(predict_from_cache(model, cache).detach().cpu().numpy().astype(np.float64))
    return np.concatenate(pieces)


@torch.no_grad()
def elr_dynamics(params: dict[str, float], curve) -> pd.DataFrame:
    model = model_from_params(params)
    cache = ELRMPLCache(curve, curve.full_steps, np.ones(len(curve.full_steps), dtype=np.float32))
    *_, C_u, n0 = model.unpack()
    norm_square, elr, s1 = cache.elr_quantities(C_u, n0)
    return pd.DataFrame(
        {
            "step": curve.full_steps,
            "lr": curve.full_lr,
            "norm_square": norm_square.detach().cpu().numpy(),
            "norm": torch.sqrt(norm_square).detach().cpu().numpy(),
            "elr": elr.detach().cpu().numpy(),
            "elr_cumsum": s1.detach().cpu().numpy(),
        }
    )


def predictions_for_all(params: dict[str, float], curves: dict[str, Any]) -> dict[str, np.ndarray]:
    return {name: predict_curve(params, curves[name]) for name in SCHEDULES}


def save_outputs(method: str, params: dict[str, Any], curves: dict[str, Any], info: dict[str, Any]) -> list[dict[str, Any]]:
    predictions = predictions_for_all(params, curves)
    rows = write_method_outputs(method, params, curves, predictions, info)
    method_dir = OUTPUT_ROOT / method
    for name in SCHEDULES:
        elr_dynamics(params, curves[name]).to_csv(method_dir / f"{name}_elr_dynamics.csv", index=False)
    return rows


def metric_value(rows: list[dict[str, Any]], schedule: str, key: str) -> float:
    for row in rows:
        if row["schedule"] == schedule and row["window"] == "post_2048":
            return float(row[key])
    raise KeyError((schedule, key))


def stable_decay_rows(method: str, curves: dict[str, Any], predictions: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows = []
    for name in SCHEDULES:
        curve = curves[name]
        boundary = detect_wsd_decay_start(curve) if name == "wsd" else int(math.floor(0.8 * int(curve.full_steps[-1])))
        for window, start, end in [
            ("stable", FIT_START_STEP, boundary - 1),
            ("decay", boundary, int(curve.full_steps[-1])),
        ]:
            mask = curve.observed_mask.copy()
            mask &= curve.full_steps >= start
            mask &= curve.full_steps <= end
            y = curve.full_loss[mask]
            pred = predictions[name][mask]
            valid = np.isfinite(y) & np.isfinite(pred)
            rows.append(
                {
                    "method": method,
                    "schedule": name,
                    "window": window,
                    "rmse": float(np.sqrt(np.mean((y[valid] - pred[valid]) ** 2))),
                }
            )
    return rows


def stage_summary(stage: str, method: str, params: dict[str, Any], rows: list[dict[str, Any]], info: dict[str, Any]) -> dict[str, Any]:
    result = {
        "stage": stage,
        "method": method,
        "fit_points": info.get("fit_points"),
        "fixed_parameters": ",".join(info.get("fixed_parameters", [])),
        "alpha": params["alpha"],
        "C_u": params["C_u"],
        "n0": params["n0"],
        "L0": params["L0"],
        "A": params["A"],
        "B": params["B"],
        "C": params["C"],
        "beta": params["beta"],
        "gamma": params["gamma"],
    }
    for schedule in SCHEDULES:
        result[f"{schedule}_rmse"] = metric_value(rows, schedule, "rmse")
        result[f"{schedule}_final_abs_error"] = metric_value(rows, schedule, "final_abs_error")
    return result


def load_initial_params() -> dict[str, float]:
    candidates = [
        OUTPUT_ROOT / "staged_multi_power_law_stage3_decay" / "params.json",
        OUTPUT_ROOT / "multi_power_law_no_sw" / "params.json",
    ]
    for path in candidates:
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            return {
                "L0": float(payload["L0"]),
                "A": float(payload["A"]),
                "alpha": float(payload["alpha"]),
                "B": float(payload["B"]),
                "C": max(float(payload["C"]), 1.0e-8),
                "beta": float(payload["beta"]),
                "gamma": float(payload["gamma"]),
                "C_u": float(payload.get("C_u", 100.0)),
                "n0": float(payload.get("n0", 10.0)),
            }
    return dict(STAGED_MPL_INITIAL_PARAMS)


def cosine_spec(start: float = 0.0, end: float = 1.0) -> FitSpec:
    return FitSpec("cosine", start_fraction=start, end_fraction=end, fit_stride=FIT_STRIDE)


def oracle_specs(decay_weight: float) -> list[FitSpec]:
    return [
        FitSpec("cosine", base_weight=1.0, wsd_decay_weight=1.0, fit_stride=FIT_STRIDE),
        FitSpec("wsd", base_weight=1.0, wsd_decay_weight=decay_weight, fit_stride=FIT_STRIDE),
    ]


def run_direct_fit(curves: dict[str, Any], initial: dict[str, float]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    config = FitConfig(
        name="direct_cosine_full",
        specs=[cosine_spec()],
        fixed=set(),
        base_only=False,
    )
    with Timer() as timer:
        params, info = fit_stage(curves, initial, config)
    info["elapsed_seconds"] = timer.elapsed
    info["method"] = f"{METHOD}_direct"
    rows = save_outputs(f"{METHOD}_direct", params, curves, info)
    return params, info, stage_summary("direct", f"{METHOD}_direct", params, rows, info), rows


def run_norm_initialization_scan(curves: dict[str, Any], base_initial: dict[str, float]) -> tuple[dict[str, Any], pd.DataFrame]:
    records = []
    payloads = []
    for C_u_init, n0_init in NORM_INITIALIZATION_GRID:
        initial = dict(base_initial)
        initial["C_u"] = C_u_init
        initial["n0"] = n0_init
        label = f"init_Cu_{C_u_init:g}_n0_{n0_init:g}".replace(".", "p").replace("+", "")
        config = FitConfig(
            name=f"norm_scan_{label}",
            specs=[cosine_spec()],
            fixed=set(),
            base_only=False,
            adam_max_steps=900,
            lbfgs_max_steps=2,
        )
        params, info = fit_stage(curves, initial, config)
        predictions = predictions_for_all(params, curves)
        metrics = compute_metric_rows(f"{METHOD}_norm_scan_{label}", curves, predictions)
        row = stage_summary("norm_init_scan", label, params, metrics, info)
        row["init_C_u"] = C_u_init
        row["init_n0"] = n0_init
        row["selection_score"] = row["wsd_rmse"] + 0.25 * row["811_rmse"]
        records.append(row)
        payloads.append((row["selection_score"], params))
        print(
            f"norm scan {label}: cosine={row['cosine_rmse']:.6f}, "
            f"WSD={row['wsd_rmse']:.6f}, 811={row['811_rmse']:.6f}, "
            f"C_u={params['C_u']:.4g}, n0={params['n0']:.4g}",
            flush=True,
        )
    frame = pd.DataFrame(records).sort_values(["selection_score", "wsd_rmse", "811_rmse"])
    frame.to_csv(OUTPUT_ROOT / f"{METHOD}_norm_initialization_scan.csv", index=False)
    selected = sorted(payloads, key=lambda item: item[0])[0][1]
    return selected, frame


def run_oracle_fit(curves: dict[str, Any], initial: dict[str, float], decay_weight: float) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    label = f"oracle_wsd_decay_x{int(decay_weight)}"
    config = FitConfig(
        name=label,
        specs=oracle_specs(decay_weight),
        fixed=set(),
        base_only=False,
        adam_max_steps=2200,
        lbfgs_max_steps=5,
    )
    with Timer() as timer:
        params, info = fit_stage(curves, initial, config)
    info["elapsed_seconds"] = timer.elapsed
    method = f"{METHOD}_{label}"
    info["method"] = method
    rows = save_outputs(method, params, curves, info)
    return params, stage_summary(f"oracle_x{int(decay_weight)}", method, params, rows, info), rows


def scan_stage1_windows(curves: dict[str, Any], initial: dict[str, float], oracle_params: dict[str, Any], label: str) -> pd.DataFrame:
    rows = []
    for fraction in STAGE1_END_FRACTIONS:
        config = FitConfig(
            name=f"stage1_base_{tag(fraction)}_{label}",
            specs=[cosine_spec(0.0, fraction)],
            fixed=set(DECAY_PARAMETERS),
            base_only=True,
            adam_max_steps=1600,
            lbfgs_max_steps=4,
        )
        params, info = fit_stage(curves, initial, config)
        predictions = predictions_for_all(params, curves)
        metrics = compute_metric_rows(f"{METHOD}_stage1_scan_{label}_{tag(fraction)}", curves, predictions)
        rows.append(
            {
                "label": label,
                "end_fraction": fraction,
                "alpha": params["alpha"],
                "oracle_alpha": oracle_params["alpha"],
                "abs_alpha_error": abs(float(params["alpha"]) - float(oracle_params["alpha"])),
                "C_u": params["C_u"],
                "n0": params["n0"],
                "L0": params["L0"],
                "A": params["A"],
                "fit_points": info["fit_points"],
                "cosine_rmse": metric_value(metrics, "cosine", "rmse"),
                "wsd_rmse": metric_value(metrics, "wsd", "rmse"),
                "811_rmse": metric_value(metrics, "811", "rmse"),
            }
        )
    frame = pd.DataFrame(rows).sort_values(["abs_alpha_error", "wsd_rmse"])
    return frame


def fit_stage1_selected(curves: dict[str, Any], initial: dict[str, float], fraction: float, label: str) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    config = FitConfig(
        name=f"stage1_base_{tag(fraction)}_{label}",
        specs=[cosine_spec(0.0, fraction)],
        fixed=set(DECAY_PARAMETERS),
        base_only=True,
        adam_max_steps=1800,
        lbfgs_max_steps=4,
    )
    params, info = fit_stage(curves, initial, config)
    method = f"{METHOD}_stage1_{label}_base_{tag(fraction)}"
    info["method"] = method
    rows = save_outputs(method, params, curves, info)
    return params, info, rows


def stage2_variants(curves: dict[str, Any], initial: dict[str, Any], label: str) -> list[tuple[str, dict[str, Any], dict[str, Any], list[dict[str, Any]]]]:
    variants = [
        ("fixed_alpha", {"alpha"}, 0.0),
        ("fixed_alpha_norm", {"alpha", "C_u", "n0"}, 0.0),
        ("fixed_base", {"L0", "A", "alpha"}, 0.0),
        ("fixed_base_norm", {"L0", "A", "alpha", "C_u", "n0"}, 0.0),
    ]
    results = []
    for name, fixed, reg in variants:
        config = FitConfig(
            name=f"stage2_{label}_{name}",
            specs=[cosine_spec()],
            fixed=set(fixed),
            parameter_regularization=reg,
            adam_max_steps=2200,
            lbfgs_max_steps=5,
        )
        params, info = fit_stage(curves, initial, config)
        method = f"{METHOD}_stage2_{label}_{name}"
        info["method"] = method
        rows = save_outputs(method, params, curves, info)
        results.append((method, params, info, rows))
    return results


def stage3_candidates(curves: dict[str, Any], stage2_params: dict[str, Any], label: str) -> list[tuple[str, dict[str, Any], dict[str, Any], list[dict[str, Any]]]]:
    fixed_options = [
        ("fixed_base", {"L0", "A", "alpha"}),
        ("fixed_base_norm", {"L0", "A", "alpha", "C_u", "n0"}),
        ("fixed_alpha_norm", {"alpha", "C_u", "n0"}),
    ]
    results = []
    for start in STAGE3_START_FRACTIONS:
        for fixed_name, fixed in fixed_options:
            config = FitConfig(
                name=f"stage3_{label}_{tag(start)}_{fixed_name}",
                specs=[cosine_spec(start, 1.0)],
                fixed=set(fixed),
                adam_max_steps=1800,
                lbfgs_max_steps=4,
            )
            params, info = fit_stage(curves, stage2_params, config)
            method = f"{METHOD}_stage3_{label}_{tag(start)}_{fixed_name}"
            info["method"] = method
            rows = save_outputs(method, params, curves, info)
            results.append((method, params, info, rows))
    return results


def reference_mpl_metrics() -> dict[str, float] | None:
    path = OUTPUT_ROOT / "staged_multi_power_law_stage3_decay" / "metrics.csv"
    if path.exists():
        frame = pd.read_csv(path)
        wsd = frame[(frame["schedule"] == "wsd") & (frame["window"] == "post_2048")]
        eight = frame[(frame["schedule"] == "811") & (frame["window"] == "post_2048")]
        if not wsd.empty and not eight.empty:
            return {
                "wsd_rmse": float(wsd.iloc[0]["rmse"]),
                "811_rmse": float(eight.iloc[0]["rmse"]),
            }
    return dict(REFERENCE_STAGED_MPL_METRICS)


def choose_final_candidate(
    method_rows: list[tuple[str, dict[str, Any], dict[str, Any], list[dict[str, Any]]]],
) -> tuple[str, dict[str, Any], dict[str, Any], list[dict[str, Any]], float, str]:
    reference = reference_mpl_metrics()
    candidates = []
    for method, params, info, rows in method_rows:
        wsd_post = metric_value(rows, "wsd", "rmse")
        eight_post = metric_value(rows, "811", "rmse")
        mean_post = 0.5 * (wsd_post + eight_post)
        max_post = max(wsd_post, eight_post)
        beats_reference = False
        if reference is not None:
            beats_reference = wsd_post <= reference["wsd_rmse"] and eight_post <= reference["811_rmse"]
        candidates.append((beats_reference, mean_post, max_post, method, params, info, rows))

    if not candidates:
        raise RuntimeError("No candidate to select.")

    beating = [item for item in candidates if item[0]]
    if beating:
        selected = sorted(beating, key=lambda item: (item[1], item[2]))[0]
        rule = "among candidates beating reference MPL 3-stage on both WSD and 811 post-RMSE, choose minimum mean post-RMSE"
    else:
        selected = sorted(candidates, key=lambda item: (item[2], item[1]))[0]
        rule = "no candidate beats reference MPL 3-stage on both WSD and 811; choose minimum max(WSD,811) post-RMSE"
    _, mean_post, _, method, params, info, rows = selected
    return method, params, info, rows, mean_post, rule


def run_pipeline() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    curves, diagnostics = load_curves()
    save_json(OUTPUT_ROOT / "data_diagnostics.json", diagnostics)

    initial = load_initial_params()
    all_rows = []
    scan_frames = []

    initial, init_scan = run_norm_initialization_scan(curves, initial)

    direct_params, direct_info, direct_summary, direct_rows = run_direct_fit(curves, initial)
    all_rows.append(direct_summary)
    initial = direct_params

    oracle_records = []
    oracle_payloads = []
    for decay_weight in ORACLE_DECAY_WEIGHTS:
        params, summary, _ = run_oracle_fit(curves, initial, decay_weight)
        all_rows.append(summary)
        oracle_records.append({"decay_weight": decay_weight, **params, **summary})
        oracle_payloads.append((decay_weight, params, summary))

    oracle_frame = pd.DataFrame(oracle_records).sort_values(["wsd_rmse", "811_rmse", "wsd_final_abs_error"])
    oracle_frame.to_csv(OUTPUT_ROOT / f"{METHOD}_oracle_summary.csv", index=False)
    selected_decay_weight = float(oracle_frame.iloc[0]["decay_weight"])
    selected_oracle = [item for item in oracle_payloads if item[0] == selected_decay_weight][0][1]

    for decay_weight, oracle_params, _ in oracle_payloads:
        label = f"w{int(decay_weight)}"
        scan = scan_stage1_windows(curves, initial, oracle_params, label)
        scan_frames.append(scan)
    scan_all = pd.concat(scan_frames, ignore_index=True)
    scan_all.to_csv(OUTPUT_ROOT / f"{METHOD}_stage1_window_scan.csv", index=False)

    selected_scan = scan_all[scan_all["label"] == f"w{int(selected_decay_weight)}"].sort_values(["abs_alpha_error", "wsd_rmse"]).iloc[0]
    stage1_fraction = float(selected_scan["end_fraction"])
    label = f"w{int(selected_decay_weight)}_base_{tag(stage1_fraction)}"
    stage1_params, stage1_info, stage1_rows = fit_stage1_selected(curves, initial, stage1_fraction, label)
    all_rows.append(stage_summary("stage1", f"{METHOD}_stage1_{label}_base_{tag(stage1_fraction)}", stage1_params, stage1_rows, stage1_info))

    stage2_initial = dict(initial)
    for name in BASE_PARAMETERS:
        stage2_initial[name] = float(stage1_params[name])
    # Oracle norm initialization is useful when the cosine prefix cannot identify C_u/n0.
    oracle_norm_initial = dict(stage2_initial)
    oracle_norm_initial["C_u"] = float(selected_oracle["C_u"])
    oracle_norm_initial["n0"] = float(selected_oracle["n0"])

    stage2_results = []
    for init_label, init_params in [("stage1_norm", stage2_initial), ("oracle_norm", oracle_norm_initial)]:
        for method, params, info, rows in stage2_variants(curves, init_params, f"{label}_{init_label}"):
            all_rows.append(stage_summary("stage2", method, params, rows, info))
            stage2_results.append((method, params, info, rows))

    stage2_best = sorted(stage2_results, key=lambda item: (metric_value(item[3], "wsd", "rmse"), metric_value(item[3], "811", "rmse")))[0]
    stage3_results = stage3_candidates(curves, stage2_best[1], f"{label}_from_best_stage2")
    for method, params, info, rows in stage3_results:
        all_rows.append(stage_summary("stage3", method, params, rows, info))

    final_candidates = [(f"{METHOD}_direct", direct_params, direct_info, direct_rows)] + stage2_results + stage3_results
    best_method, best_params, best_info, best_rows, best_score, selection_rule = choose_final_candidate(final_candidates)
    final_method = f"{METHOD}_final"
    final_info = {
        "method": final_method,
        "source_candidate_method": best_method,
        "source_stage2_method": stage2_best[0],
        "reference_mpl_metrics": reference_mpl_metrics(),
        "selection_rule": selection_rule,
        "selection_score": best_score,
        "norm_initialization_scan": str(OUTPUT_ROOT / f"{METHOD}_norm_initialization_scan.csv"),
        "selected_norm_initialization": {
            "init_C_u": float(init_scan.iloc[0]["init_C_u"]),
            "init_n0": float(init_scan.iloc[0]["init_n0"]),
            "C_u": float(init_scan.iloc[0]["C_u"]),
            "n0": float(init_scan.iloc[0]["n0"]),
        },
        "selected_oracle_decay_weight": selected_decay_weight,
        "selected_stage1_fraction": stage1_fraction,
        "optimizer": "All ELR-MPL stages use Adam/LBFGS early stopping, not fixed iteration budgets.",
        "selected_candidate_info": best_info,
    }
    final_rows = save_outputs(final_method, best_params, curves, final_info)
    all_rows.append(stage_summary("final", final_method, best_params, final_rows, final_info))

    summary = pd.DataFrame(all_rows)
    summary.to_csv(OUTPUT_ROOT / f"{METHOD}_summary.csv", index=False)
    save_json(
        OUTPUT_ROOT / f"{METHOD}_summary.json",
        {
            "method": METHOD,
            "fit_stride": FIT_STRIDE,
            "oracle_decay_weights": ORACLE_DECAY_WEIGHTS,
            "norm_initialization_grid": NORM_INITIALIZATION_GRID,
            "norm_initialization_scan_csv": str(OUTPUT_ROOT / f"{METHOD}_norm_initialization_scan.csv"),
            "stage1_end_fractions": STAGE1_END_FRACTIONS,
            "stage3_start_fractions": STAGE3_START_FRACTIONS,
            "selected_oracle_decay_weight": selected_decay_weight,
            "selected_stage1_fraction": stage1_fraction,
            "final_method": final_method,
            "final_params": best_params,
            "summary_csv": str(OUTPUT_ROOT / f"{METHOD}_summary.csv"),
            "oracle_summary_csv": str(OUTPUT_ROOT / f"{METHOD}_oracle_summary.csv"),
            "stage1_window_scan_csv": str(OUTPUT_ROOT / f"{METHOD}_stage1_window_scan.csv"),
        },
    )
    print(summary.sort_values(["stage", "wsd_rmse", "811_rmse"]).to_string(index=False))
    print("Final params:")
    print(pd.Series(best_params).to_string())


if __name__ == "__main__":
    apply_runtime_args(parse_args())
    with Timer() as total_timer:
        run_pipeline()
    print(f"Staged ELR-MPL pipeline finished in {total_timer.elapsed:.2f}s")
