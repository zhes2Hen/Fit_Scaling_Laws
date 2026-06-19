import argparse
import copy
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
    write_data_diagnostics,
    write_method_outputs,
)


METHOD = "staged_elr_functional_scaling_law"
BASELINE_SOURCE = "embedded lambda-zero ELR-FSL initialization from the best previous run"
DEFAULT_BASELINE = {
    "c1": 7999.89794921875,
    "c2": 1.1852997541427612,
    "c3": 94.8235092163086,
    "c4": 5000.0185546875,
    "beta": 2.0300283432006836,
    "s": 0.8211127519607544,
    "sigma": 0.17647188901901245,
    "C_u": 9.999651229009032e-05,
    "n0": 0.9926978349685669,
}

FIT_START_STEP = PRIMARY_EVAL_START
FIT_STRIDE = 20
M_WIDTH = 128.0
NOISE_POINTS = 384
E_POINTS = 161
LAMBDA_WD = 0.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32

PARAMETERS = ("c1", "c2", "c3", "c4", "beta", "s", "sigma", "C_u", "n0")
OUTPUT_KEYS = {
    "c1": "c1_constant_model_term",
    "c2": "c2_signal_term",
    "c3": "c3_minibatch_noise_term",
    "c4": "c4_label_noise_term",
}


@dataclass(frozen=True)
class FitConfig:
    name: str
    start_fraction: float
    end_fraction: float
    fixed: set[str]
    adam_steps: int
    lbfgs_steps: int
    fit_stride: int = FIT_STRIDE
    lr: float = 2.0e-2
    regularization: float = 0.0


def inv_softplus(x: float) -> float:
    if x > 20.0:
        return x
    return math.log(math.expm1(max(x, 1.0e-8)))


def logit(x: float) -> float:
    x = min(max(x, 1.0e-6), 1.0 - 1.0e-6)
    return math.log(x / (1.0 - x))


def raw_from_value(name: str, value: float) -> float:
    if name in {"c1", "c2", "c3", "c4", "sigma", "C_u", "n0"}:
        return inv_softplus(value)
    if name == "beta":
        return logit((value - 1.0) / 5.0)
    if name == "s":
        return logit(value / 2.0)
    raise KeyError(name)


def value_from_raw(name: str, raw: torch.Tensor) -> torch.Tensor:
    if name in {"c1", "c2", "c3", "c4", "sigma"}:
        return torch.nn.functional.softplus(raw) + 1.0e-10
    if name in {"C_u", "n0"}:
        return torch.nn.functional.softplus(raw) + 1.0e-12
    if name == "beta":
        return 1.0 + 5.0 * torch.sigmoid(raw)
    if name == "s":
        return 2.0 * torch.sigmoid(raw) + 1.0e-8
    raise KeyError(name)


def normalize_params(params: dict[str, Any]) -> dict[str, float]:
    result = {
        "c1": float(params.get("c1", params.get("c1_constant_model_term"))),
        "c2": float(params.get("c2", params.get("c2_signal_term"))),
        "c3": float(params.get("c3", params.get("c3_minibatch_noise_term"))),
        "c4": float(params.get("c4", params.get("c4_label_noise_term"))),
        "beta": float(params["beta"]),
        "s": float(params["s"]),
        "sigma": float(params["sigma"]),
        "C_u": float(params["C_u"]),
        "n0": float(params["n0"]),
    }
    for name in PARAMETERS:
        if not math.isfinite(result[name]):
            raise ValueError(f"{name} is not finite: {result[name]}")
    return result


def output_params(params: dict[str, float]) -> dict[str, float | bool | str]:
    return {
        "c1_constant_model_term": float(params["c1"]),
        "c2_signal_term": float(params["c2"]),
        "c3_minibatch_noise_term": float(params["c3"]),
        "c4_label_noise_term": float(params["c4"]),
        "M": M_WIDTH,
        "beta": float(params["beta"]),
        "s": float(params["s"]),
        "sigma": float(params["sigma"]),
        "lambda_wd": LAMBDA_WD,
        "C_u": float(params["C_u"]),
        "n0": float(params["n0"]),
        "fit_lambda": False,
        "fixed_lambda": 0.0,
        "intrinsic_time": "sum eta_k / sqrt(n_k), with lambda fixed to 0",
        "norm_recurrence": "n_{k+1}=n_k+eta_k^2*C_u",
    }


class StagedELRFSLModel(torch.nn.Module):
    def __init__(self, initial: dict[str, float], fixed: set[str]):
        super().__init__()
        self.fixed = set(fixed)
        self.trainable = [name for name in PARAMETERS if name not in self.fixed]
        if not self.trainable:
            raise ValueError("At least one parameter must be trainable.")

        for name in self.fixed:
            self.register_buffer(f"fixed_{name}", torch.tensor(float(initial[name]), dtype=DTYPE, device=DEVICE))

        raw = [raw_from_value(name, float(initial[name])) for name in self.trainable]
        raw_tensor = torch.tensor(raw, dtype=DTYPE, device=DEVICE)
        self.raw = torch.nn.Parameter(raw_tensor.clone())
        self.register_buffer("raw_reference", raw_tensor.clone())

    def value(self, name: str) -> torch.Tensor:
        if name in self.fixed:
            return getattr(self, f"fixed_{name}")
        index = self.trainable.index(name)
        return value_from_raw(name, self.raw[index])

    def unpack(self) -> tuple[torch.Tensor, ...]:
        return tuple(self.value(name) for name in PARAMETERS)

    def params_dict(self) -> dict[str, float]:
        values = self.unpack()
        return {name: float(value.detach().cpu()) for name, value in zip(PARAMETERS, values)}


class ELRFSLCache:
    def __init__(self, curve, target_steps: np.ndarray):
        self.steps_np = target_steps.astype(np.float32)
        self.steps = torch.tensor(self.steps_np, dtype=DTYPE, device=DEVICE)
        self.loss = torch.tensor(curve.losses_at(target_steps), dtype=DTYPE, device=DEVICE)
        self.lr = torch.tensor(curve.full_lr.astype(np.float32), dtype=DTYPE, device=DEVICE)

    def norm_and_elr(self, C_u: torch.Tensor, n0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        eta = self.lr
        increments = eta.pow(2) * C_u
        sum_before = torch.cat(
            [
                torch.zeros(1, dtype=DTYPE, device=DEVICE),
                torch.cumsum(increments, dim=0)[:-1],
            ]
        )
        norm_square = (n0 + sum_before).clamp_min(1.0e-12)
        norm = torch.sqrt(norm_square)
        elr = eta / norm
        return norm_square, norm, elr

    def intrinsic_time(self, schedule: torch.Tensor, cumsum_schedule: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        r = r.to(device=DEVICE, dtype=DTYPE)
        floor_r = torch.floor(r).long()
        frac = r - floor_r.to(DTYPE)
        max_idx = len(schedule) - 1

        idx_complete = torch.clamp(floor_r - 1, min=0, max=max_idx)
        complete = torch.where(
            floor_r > 0,
            cumsum_schedule[idx_complete],
            torch.zeros_like(r, dtype=DTYPE, device=DEVICE),
        )
        idx_partial = torch.clamp(floor_r, min=0, max=max_idx)
        valid = floor_r <= max_idx
        partial = torch.where(valid, schedule[idx_partial] * frac, torch.zeros_like(r, dtype=DTYPE, device=DEVICE))
        return complete + partial


def simpson_e(t: torch.Tensor, beta: torch.Tensor, s: torch.Tensor, n_points: int = E_POINTS) -> torch.Tensor:
    if n_points % 2 == 0:
        n_points += 1
    base = torch.linspace(0.0, 1.0, n_points, dtype=DTYPE, device=DEVICE)
    lower = torch.pow(torch.tensor(M_WIDTH, dtype=DTYPE, device=DEVICE), -beta)
    z = lower + (1.0 - lower) * base
    dz = (1.0 - lower) / (n_points - 1)
    weights = torch.ones(n_points, dtype=DTYPE, device=DEVICE)
    weights[1:-1:2] = 4.0
    weights[2:-1:2] = 2.0
    log_values = (s - 1.0) * torch.log(z[:, None].clamp_min(1.0e-12)) - 2.0 * z[:, None] * t[None, :]
    values = torch.exp(log_values)
    return (weights[:, None] * values).sum(dim=0) * dz / 3.0


def forgetting_kernel(t: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    return (1.0 + t.clamp_min(0.0)).pow(-2.0 + 1.0 / beta)


def expected_terms(cache: ELRFSLCache, beta, s, sigma, C_u, n0) -> torch.Tensor:
    _, _, elr = cache.norm_and_elr(C_u, n0)
    cumsum_elr = torch.cumsum(elr, dim=0)

    max_step = float(torch.max(cache.steps).detach().cpu())
    r = torch.linspace(0.0, max_step, NOISE_POINTS, dtype=DTYPE, device=DEVICE)
    dr = r[1] - r[0]

    t_intrinsic = cache.intrinsic_time(elr, cumsum_elr, cache.steps)
    r_intrinsic = cache.intrinsic_time(elr, cumsum_elr, r)
    diff = t_intrinsic[:, None] - r_intrinsic[None, :]
    kernel = torch.where(diff > 0, forgetting_kernel(diff, beta), torch.zeros_like(diff))

    e_values = simpson_e(r_intrinsic, beta, s)
    idx_r = torch.clamp(torch.floor(r).long(), max=len(elr) - 1)
    elr_sq = elr[idx_r].pow(2)

    integrand_mini = kernel * e_values[None, :] * elr_sq[None, :]
    noise_mini = (integrand_mini[:, 1:] + integrand_mini[:, :-1]).sum(dim=1) * dr / 2.0

    integrand_label = kernel * sigma.pow(2) * elr_sq[None, :]
    noise_label = (integrand_label[:, 1:] + integrand_label[:, :-1]).sum(dim=1) * dr / 2.0

    term1_value = torch.pow(torch.tensor(M_WIDTH, dtype=DTYPE, device=DEVICE), -s * beta)
    term1 = torch.ones_like(cache.steps) * term1_value
    term2 = t_intrinsic.clamp_min(1.0e-8).pow(-s)
    return torch.stack([term1, term2, noise_mini, noise_label], dim=1)


def predict_from_cache(model: StagedELRFSLModel, cache: ELRFSLCache) -> torch.Tensor:
    c1, c2, c3, c4, beta, s, sigma, C_u, n0 = model.unpack()
    terms = expected_terms(cache, beta, s, sigma, C_u, n0)
    coeffs = torch.stack([c1, c2, c3, c4])
    return terms @ coeffs


def log_huber(prediction: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
    residual = torch.log(observed) - torch.log(prediction.clamp_min(1.0e-8))
    abs_r = torch.abs(residual)
    return torch.where(
        abs_r <= HUBER_DELTA,
        0.5 * residual * residual,
        HUBER_DELTA * (abs_r - 0.5 * HUBER_DELTA),
    ).mean()


def training_loss(model: StagedELRFSLModel, cache: ELRFSLCache, regularization: float) -> torch.Tensor:
    prediction = predict_from_cache(model, cache)
    penalty = torch.relu(1.0e-6 - prediction).pow(2).mean() * 1.0e6
    drift = float(regularization) * (model.raw - model.raw_reference).pow(2).mean()
    return log_huber(prediction, cache.loss) + penalty + drift


def fit_steps(curve, start_fraction: float, end_fraction: float, stride: int) -> np.ndarray:
    if not 0.0 <= start_fraction < 1.0:
        raise ValueError("start_fraction must be in [0, 1).")
    if not 0.0 < end_fraction <= 1.0:
        raise ValueError("end_fraction must be in (0, 1].")
    if start_fraction >= end_fraction:
        raise ValueError("start_fraction must be smaller than end_fraction.")

    last_step = int(curve.full_steps[-1])
    start = max(FIT_START_STEP, int(math.floor(start_fraction * last_step)))
    end = int(math.floor(end_fraction * last_step))
    keep = (curve.observed_steps >= start) & (curve.observed_steps <= end)
    steps = curve.observed_steps[keep]
    steps = steps[(steps - start) % stride == 0]
    if len(steps) == 0 or steps[-1] != curve.observed_steps[keep][-1]:
        steps = np.append(steps, curve.observed_steps[keep][-1])
    return np.unique(steps.astype(np.int64))


def fit_one_stage(initial: dict[str, float], curve, config: FitConfig) -> tuple[dict[str, float], dict[str, Any]]:
    steps = fit_steps(curve, config.start_fraction, config.end_fraction, config.fit_stride)
    cache = ELRFSLCache(curve, steps)
    model = StagedELRFSLModel(initial, config.fixed).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    history = []

    for step in range(config.adam_steps):
        optimizer.zero_grad(set_to_none=True)
        value = training_loss(model, cache, config.regularization)
        value.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()

        scalar = float(value.detach().cpu())
        if step % 50 == 0 or step == config.adam_steps - 1:
            history.append({"step": step, "loss": scalar})
        if math.isfinite(scalar) and scalar < best_loss:
            best_loss = scalar
            best_state = copy.deepcopy(model.state_dict())
        if step % 400 == 0:
            print(f"{config.name} Adam {step:4d}: {scalar:.6e}, best={best_loss:.6e}")

    model.load_state_dict(best_state)
    if config.lbfgs_steps > 0:
        lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=0.7,
            max_iter=config.lbfgs_steps,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            lbfgs.zero_grad(set_to_none=True)
            value = training_loss(model, cache, config.regularization)
            value.backward()
            return value

        lbfgs.step(closure)
        final = float(training_loss(model, cache, config.regularization).detach().cpu())
        if math.isfinite(final) and final < best_loss:
            best_loss = final
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    params = model.params_dict()
    info = {
        "stage_name": config.name,
        "fixed_parameters": sorted(config.fixed),
        "trainable_parameters": [name for name in PARAMETERS if name not in config.fixed],
        "start_fraction": config.start_fraction,
        "end_fraction": config.end_fraction,
        "fit_stride": config.fit_stride,
        "fit_points": int(len(steps)),
        "fit_step_min": int(steps.min()),
        "fit_step_max": int(steps.max()),
        "adam_steps": config.adam_steps,
        "lbfgs_steps": config.lbfgs_steps,
        "regularization": config.regularization,
        "best_training_loss": best_loss,
        "history": history,
    }
    return params, info


@torch.no_grad()
def predict_curve(params: dict[str, float], curve) -> np.ndarray:
    model = StagedELRFSLModel(params, set(PARAMETERS) - {"c1"}).to(DEVICE)
    model.fixed = set(PARAMETERS)
    for name in PARAMETERS:
        if hasattr(model, f"fixed_{name}"):
            getattr(model, f"fixed_{name}").fill_(float(params[name]))
        else:
            model.register_buffer(f"fixed_{name}", torch.tensor(float(params[name]), dtype=DTYPE, device=DEVICE))
    cache = ELRFSLCache(curve, curve.full_steps)
    prediction = predict_from_cache(model, cache)
    return prediction.detach().cpu().numpy().astype(np.float64)


def predictions_for_all(params: dict[str, float], curves) -> dict[str, np.ndarray]:
    return {name: predict_curve(params, curves[name]) for name in SCHEDULES}


def save_stage(method: str, params: dict[str, float], curves, info: dict[str, Any]) -> list[dict[str, Any]]:
    predictions = predictions_for_all(params, curves)
    payload = {
        **output_params(params),
        "staged_raw_params": params,
    }
    return write_method_outputs(method, payload, curves, predictions, info)


def metric_value(rows: list[dict[str, Any]], schedule: str, window: str, key: str = "rmse") -> float:
    for row in rows:
        if row["schedule"] == schedule and row["window"] == window:
            return float(row[key])
    raise KeyError((schedule, window, key))


def load_baseline() -> dict[str, float]:
    return dict(DEFAULT_BASELINE)


def tag(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def stage_summary(
    stage: str,
    method: str,
    params: dict[str, float],
    rows: list[dict[str, Any]],
    info: dict[str, Any],
) -> dict[str, Any]:
    result = {
        "stage": stage,
        "method": method,
        "fit_points": info.get("fit_points"),
        "start_fraction": info.get("start_fraction"),
        "fixed_parameters": ",".join(info.get("fixed_parameters", [])),
        "c1": params["c1"],
        "c2": params["c2"],
        "c3": params["c3"],
        "c4": params["c4"],
        "beta": params["beta"],
        "s": params["s"],
        "sigma": params["sigma"],
        "C_u": params["C_u"],
        "n0": params["n0"],
    }
    for schedule in SCHEDULES:
        result[f"{schedule}_post_2048_rmse"] = metric_value(rows, schedule, "post_2048", "rmse")
        result[f"{schedule}_final_abs_error"] = metric_value(rows, schedule, "post_2048", "final_abs_error")
    return result


ORACLE_DECAY_WEIGHTS = [10.0, 20.0]
STAGE1_END_FRACTIONS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60]
STAGE3_STARTS = [0.925, 0.935, 0.94, 0.945, 0.955]
STAGE3_VARIANTS = {
    "noise_only": {"c1", "c2", "beta", "s", "C_u", "n0"},
    "c2_free": {"c1", "beta", "s", "C_u", "n0"},
    "beta_free": {"c1", "c2", "s", "C_u", "n0"},
    "c2_beta_free": {"c1", "s", "C_u", "n0"},
}


@dataclass(frozen=True)
class WeightedSpec:
    schedule: str
    start_fraction: float
    end_fraction: float
    base_weight: float = 1.0
    wsd_decay_weight: float = 1.0
    fit_stride: int = 20


@dataclass(frozen=True)
class WeightedFitConfig:
    name: str
    fixed: set[str]
    specs: list[WeightedSpec]
    adam_steps: int
    lbfgs_steps: int
    lr: float = 2.0e-2
    regularization: float = 0.0


class WeightedCache:
    def __init__(self, curve, steps: np.ndarray, weights: np.ndarray):
        self.curve_name = curve.name
        self.cache = ELRFSLCache(curve, steps)
        self.weights = torch.tensor(weights.astype(np.float32), dtype=DTYPE, device=DEVICE)
        self.fit_points = int(len(steps))
        self.step_min = int(steps.min())
        self.step_max = int(steps.max())


def parse_float_list(text: str) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated number.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run staged ELR-FSL with WSD-validation settings.")
    parser.add_argument("--fit-stride", type=int, default=20)
    parser.add_argument("--oracle-decay-weights", type=parse_float_list, default=ORACLE_DECAY_WEIGHTS)
    parser.add_argument("--stage1-end-fractions", type=parse_float_list, default=STAGE1_END_FRACTIONS)
    parser.add_argument("--stage3-starts", type=parse_float_list, default=STAGE3_STARTS)
    return parser.parse_args()


def apply_runtime_args(args: argparse.Namespace) -> None:
    global FIT_STRIDE, ORACLE_DECAY_WEIGHTS, STAGE1_END_FRACTIONS, STAGE3_STARTS
    if args.fit_stride <= 0:
        raise ValueError("--fit-stride must be positive.")
    FIT_STRIDE = int(args.fit_stride)
    ORACLE_DECAY_WEIGHTS = list(args.oracle_decay_weights)
    STAGE1_END_FRACTIONS = list(args.stage1_end_fractions)
    STAGE3_STARTS = list(args.stage3_starts)


def first_decay_step(curve) -> int | None:
    drops = np.where(np.diff(curve.full_lr) < -1.0e-12)[0] + 1
    if len(drops) == 0:
        return None
    return int(drops[0])


def weights_for_steps(curve, steps: np.ndarray, spec: WeightedSpec) -> np.ndarray:
    weights = np.full(len(steps), float(spec.base_weight), dtype=np.float64)
    decay_start = first_decay_step(curve)
    if curve.name == "wsd" and decay_start is not None:
        weights[steps >= decay_start] = float(spec.wsd_decay_weight)
    return weights


def weighted_specs_for_decay(decay_weight: float) -> list[WeightedSpec]:
    return [
        WeightedSpec("cosine", 0.0, 1.0, base_weight=1.0, wsd_decay_weight=1.0, fit_stride=FIT_STRIDE),
        WeightedSpec("wsd", 0.0, 1.0, base_weight=1.0, wsd_decay_weight=decay_weight, fit_stride=FIT_STRIDE),
    ]


def weighted_training_loss(
    model: StagedELRFSLModel,
    weighted_caches: list[WeightedCache],
    regularization: float,
) -> torch.Tensor:
    numerator = torch.zeros((), dtype=DTYPE, device=DEVICE)
    denominator = torch.zeros((), dtype=DTYPE, device=DEVICE)
    penalty = torch.zeros((), dtype=DTYPE, device=DEVICE)
    for item in weighted_caches:
        prediction = predict_from_cache(model, item.cache)
        residual = torch.log(item.cache.loss) - torch.log(prediction.clamp_min(1.0e-8))
        abs_r = torch.abs(residual)
        loss = torch.where(
            abs_r <= HUBER_DELTA,
            0.5 * residual * residual,
            HUBER_DELTA * (abs_r - 0.5 * HUBER_DELTA),
        )
        numerator = numerator + (loss * item.weights).sum()
        denominator = denominator + item.weights.sum()
        penalty = penalty + torch.relu(1.0e-6 - prediction).pow(2).mean() * 1.0e6
    drift = float(regularization) * (model.raw - model.raw_reference).pow(2).mean()
    return numerator / denominator.clamp_min(1.0e-12) + penalty / len(weighted_caches) + drift


def fit_weighted_stage(
    initial: dict[str, float],
    curves,
    config: WeightedFitConfig,
) -> tuple[dict[str, float], dict[str, Any]]:
    weighted_caches: list[WeightedCache] = []
    spec_info = []
    for spec in config.specs:
        curve = curves[spec.schedule]
        steps = fit_steps(curve, spec.start_fraction, spec.end_fraction, spec.fit_stride)
        weights = weights_for_steps(curve, steps, spec)
        weighted_caches.append(WeightedCache(curve, steps, weights))
        spec_info.append(
            {
                "schedule": spec.schedule,
                "start_fraction": spec.start_fraction,
                "end_fraction": spec.end_fraction,
                "fit_stride": spec.fit_stride,
                "fit_points": int(len(steps)),
                "fit_step_min": int(steps.min()),
                "fit_step_max": int(steps.max()),
                "base_weight": spec.base_weight,
                "wsd_decay_weight": spec.wsd_decay_weight,
                "decay_start": first_decay_step(curve),
            }
        )

    model = StagedELRFSLModel(initial, config.fixed).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    history = []

    for step in range(config.adam_steps):
        optimizer.zero_grad(set_to_none=True)
        value = weighted_training_loss(model, weighted_caches, config.regularization)
        value.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()

        scalar = float(value.detach().cpu())
        if step % 50 == 0 or step == config.adam_steps - 1:
            history.append({"step": step, "loss": scalar})
        if math.isfinite(scalar) and scalar < best_loss:
            best_loss = scalar
            best_state = copy.deepcopy(model.state_dict())
        if step % 400 == 0:
            print(f"{config.name} Adam {step:4d}: {scalar:.6e}, best={best_loss:.6e}")

    model.load_state_dict(best_state)
    if config.lbfgs_steps > 0:
        lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=0.7,
            max_iter=config.lbfgs_steps,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            lbfgs.zero_grad(set_to_none=True)
            value = weighted_training_loss(model, weighted_caches, config.regularization)
            value.backward()
            return value

        lbfgs.step(closure)
        final = float(weighted_training_loss(model, weighted_caches, config.regularization).detach().cpu())
        if math.isfinite(final) and final < best_loss:
            best_loss = final
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    params = model.params_dict()
    info = {
        "stage_name": config.name,
        "fixed_parameters": sorted(config.fixed),
        "trainable_parameters": [name for name in PARAMETERS if name not in config.fixed],
        "adam_steps": config.adam_steps,
        "lbfgs_steps": config.lbfgs_steps,
        "regularization": config.regularization,
        "best_training_loss": best_loss,
        "history": history,
        "weighted_specs": spec_info,
        "fit_points": int(sum(item.fit_points for item in weighted_caches)),
        "start_fraction": None,
    }
    return params, info


def stage1_config(end_fraction: float, oracle_tag: str) -> FitConfig:
    return FitConfig(
        name=f"stage1_base_{tag(end_fraction)}_{oracle_tag}",
        start_fraction=0.0,
        end_fraction=end_fraction,
        fixed={"c3", "c4", "beta", "sigma", "C_u", "n0"},
        adam_steps=1400,
        lbfgs_steps=60,
        fit_stride=FIT_STRIDE,
    )


def stage2_config(label: str) -> FitConfig:
    return FitConfig(
        name=f"stage2_full_fixed_s_{label}",
        start_fraction=0.0,
        end_fraction=1.0,
        fixed={"s"},
        adam_steps=1600,
        lbfgs_steps=70,
        fit_stride=FIT_STRIDE,
    )


def stage3_configs(label: str) -> list[FitConfig]:
    configs = []
    for variant, fixed in STAGE3_VARIANTS.items():
        for start in STAGE3_STARTS:
            configs.append(
                FitConfig(
                    name=f"stage3_tail_{tag(start)}_{variant}_{label}",
                    start_fraction=start,
                    end_fraction=1.0,
                    fixed=set(fixed),
                    adam_steps=1100,
                    lbfgs_steps=50,
                    fit_stride=FIT_STRIDE,
                )
            )
    return configs


def save_method_outputs(method: str, params: dict[str, float], curves, info: dict[str, Any]) -> list[dict[str, Any]]:
    predictions = predictions_for_all(params, curves)
    payload = {
        **output_params(params),
        "staged_raw_params": params,
    }
    return write_method_outputs(method, payload, curves, predictions, info)


def fit_oracle_s(curves, baseline: dict[str, float], decay_weight: float) -> tuple[dict[str, float], dict[str, Any], list[dict[str, Any]]]:
    config = WeightedFitConfig(
        name=f"oracle_s_wsd_decay_x{int(decay_weight)}",
        fixed=set(),
        specs=weighted_specs_for_decay(decay_weight),
        adam_steps=1800,
        lbfgs_steps=80,
    )
    params, info = fit_weighted_stage(baseline, curves, config)
    method = f"{METHOD}_oracle_s_wsd_decay_x{int(decay_weight)}"
    rows = save_method_outputs(method, params, curves, {"method": method, **info})
    return params, info, rows


def run_stage1_window_scan(curves, baseline: dict[str, float], oracle_params: dict[str, float], oracle_label: str) -> pd.DataFrame:
    cosine = curves["cosine"]
    rows = []
    for end_fraction in STAGE1_END_FRACTIONS:
        initial = dict(baseline)
        initial["c3"] = 0.0
        initial["c4"] = 0.0
        params, info = fit_one_stage(initial, cosine, stage1_config(end_fraction, oracle_label))
        predictions = predictions_for_all(params, curves)
        metrics = compute_metric_rows(f"{METHOD}_scan_{oracle_label}_{tag(end_fraction)}", curves, predictions)
        row = {
            "oracle_label": oracle_label,
            "end_fraction": end_fraction,
            "stage1_s": params["s"],
            "oracle_s": oracle_params["s"],
            "abs_s_error": abs(params["s"] - oracle_params["s"]),
            "stage1_c1": params["c1"],
            "stage1_c2": params["c2"],
            "cosine_rmse": metric_value(metrics, "cosine", "post_2048", "rmse"),
            "wsd_rmse": metric_value(metrics, "wsd", "post_2048", "rmse"),
            "811_rmse": metric_value(metrics, "811", "post_2048", "rmse"),
            "fit_points": info["fit_points"],
        }
        rows.append(row)
        print(row)
    return pd.DataFrame(rows).sort_values(["abs_s_error", "end_fraction"])


def run_staged_pipeline(
    curves,
    baseline: dict[str, float],
    stage1_params: dict[str, float],
    label: str,
    oracle_source: dict[str, Any],
) -> list[dict[str, Any]]:
    cosine = curves["cosine"]
    summary_rows = []

    stage1_method = f"{METHOD}_stage1_{label}"
    stage1_info = {
        "method": stage1_method,
        "stage_name": "stage1_selected_oracle_s",
        "oracle_source": oracle_source,
        "fixed_parameters": ["C_u", "beta", "c3", "c4", "n0", "sigma"],
        "fit_points": oracle_source.get("fit_points"),
        "start_fraction": 0.0,
    }
    rows = save_method_outputs(stage1_method, stage1_params, curves, stage1_info)
    summary_rows.append(stage_summary("stage1", stage1_method, stage1_params, rows, stage1_info))

    stage2_initial = dict(baseline)
    stage2_initial["c1"] = stage1_params["c1"]
    stage2_initial["c2"] = stage1_params["c2"]
    stage2_initial["s"] = stage1_params["s"]
    stage2_params, stage2_info = fit_one_stage(stage2_initial, cosine, stage2_config(label))
    stage2_method = f"{METHOD}_stage2_{label}"
    rows = save_method_outputs(stage2_method, stage2_params, curves, {"method": stage2_method, **stage2_info, "oracle_source": oracle_source})
    summary_rows.append(stage_summary("stage2", stage2_method, stage2_params, rows, stage2_info))

    best_score = float("inf")
    best_payload = None
    for config in stage3_configs(label):
        stage3_params, stage3_info = fit_one_stage(stage2_params, cosine, config)
        method = f"{METHOD}_{config.name}"
        rows = save_method_outputs(method, stage3_params, curves, {"method": method, **stage3_info, "oracle_source": oracle_source})
        summary = stage_summary("stage3", method, stage3_params, rows, stage3_info)
        summary_rows.append(summary)
        score = summary["wsd_post_2048_rmse"]
        if score < best_score:
            best_score = score
            best_payload = (method, stage3_params, rows, stage3_info)

    if best_payload is None:
        raise RuntimeError(f"No Stage 3 candidate for {label}.")

    best_method, best_params, _, best_info = best_payload
    final_method = f"{METHOD}_final_{label}"
    final_info = {
        "method": final_method,
        "source_stage3_method": best_method,
        "selection_rule": "minimum WSD post-2048 RMSE among Stage-3 candidates",
        "oracle_source": oracle_source,
        "stage3_selected": best_info,
    }
    rows = save_method_outputs(final_method, best_params, curves, final_info)
    summary_rows.append(stage_summary("final", final_method, best_params, rows, final_info))
    return summary_rows


def run_oracle_initialized_pipeline(
    curves,
    oracle_params: dict[str, float],
    label: str,
    fixed_stage2: set[str],
    oracle_source: dict[str, Any],
) -> list[dict[str, Any]]:
    cosine = curves["cosine"]
    summary_rows = []

    oracle_method = f"{METHOD}_oracle_initial_{label}"
    oracle_info = {
        "method": oracle_method,
        "stage_name": "oracle_initial_parameters",
        "oracle_source": oracle_source,
        "fixed_parameters": [],
        "fit_points": None,
        "start_fraction": None,
    }
    rows = save_method_outputs(oracle_method, oracle_params, curves, oracle_info)
    summary_rows.append(stage_summary("oracle_initial", oracle_method, oracle_params, rows, oracle_info))

    stage2_params, stage2_info = fit_one_stage(
        oracle_params,
        cosine,
        FitConfig(
            name=f"stage2_oracle_init_{label}",
            start_fraction=0.0,
            end_fraction=1.0,
            fixed=set(fixed_stage2),
            adam_steps=1400,
            lbfgs_steps=60,
            fit_stride=FIT_STRIDE,
        ),
    )
    stage2_method = f"{METHOD}_stage2_oracle_init_{label}"
    rows = save_method_outputs(stage2_method, stage2_params, curves, {"method": stage2_method, **stage2_info, "oracle_source": oracle_source})
    summary_rows.append(stage_summary("stage2", stage2_method, stage2_params, rows, stage2_info))

    best_score = float("inf")
    best_payload = None
    for config in stage3_configs(label):
        stage3_params, stage3_info = fit_one_stage(stage2_params, cosine, config)
        method = f"{METHOD}_{config.name}"
        rows = save_method_outputs(method, stage3_params, curves, {"method": method, **stage3_info, "oracle_source": oracle_source})
        summary = stage_summary("stage3", method, stage3_params, rows, stage3_info)
        summary_rows.append(summary)
        score = summary["wsd_post_2048_rmse"]
        if score < best_score:
            best_score = score
            best_payload = (method, stage3_params, rows, stage3_info)

    if best_payload is None:
        raise RuntimeError(f"No oracle-initialized Stage 3 candidate for {label}.")

    best_method, best_params, _, best_info = best_payload
    final_method = f"{METHOD}_final_{label}"
    final_info = {
        "method": final_method,
        "source_stage3_method": best_method,
        "selection_rule": "minimum WSD post-2048 RMSE among oracle-initialized Stage-3 candidates",
        "oracle_source": oracle_source,
        "stage3_selected": best_info,
    }
    rows = save_method_outputs(final_method, best_params, curves, final_info)
    summary_rows.append(stage_summary("final", final_method, best_params, rows, final_info))
    return summary_rows


def make_direct_oracle_stage1(selected_stage1: dict[str, float], oracle_params: dict[str, float]) -> dict[str, float]:
    direct = dict(selected_stage1)
    direct["s"] = oracle_params["s"]
    return direct


def run_pipeline(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = parse_args()
    apply_runtime_args(args)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    curves, diagnostics = load_curves()
    write_data_diagnostics(diagnostics)
    baseline = load_baseline()

    all_summary_rows = []
    oracle_rows = []
    stage1_scan_frames = []
    with Timer() as timer:
        for decay_weight in ORACLE_DECAY_WEIGHTS:
            oracle_label = f"w{int(decay_weight)}"
            oracle_params, oracle_info, rows = fit_oracle_s(curves, baseline, decay_weight)
            oracle_summary = stage_summary(f"oracle_{oracle_label}", f"{METHOD}_oracle_s_wsd_decay_x{int(decay_weight)}", oracle_params, rows, oracle_info)
            all_summary_rows.append(oracle_summary)
            oracle_rows.append({"label": oracle_label, "decay_weight": decay_weight, **oracle_params, **oracle_summary})

            scan = run_stage1_window_scan(curves, baseline, oracle_params, oracle_label)
            stage1_scan_frames.append(scan)
            selected = scan.iloc[0].to_dict()

            selected_initial = dict(baseline)
            selected_initial["c3"] = 0.0
            selected_initial["c4"] = 0.0
            selected_stage1, selected_info = fit_one_stage(
                selected_initial,
                curves["cosine"],
                stage1_config(float(selected["end_fraction"]), f"{oracle_label}_selected"),
            )
            selected_source = {
                "oracle_label": oracle_label,
                "decay_weight": decay_weight,
                "oracle_s": oracle_params["s"],
                "stage1_end_fraction": float(selected["end_fraction"]),
                "stage1_s": selected_stage1["s"],
                "abs_s_error": abs(selected_stage1["s"] - oracle_params["s"]),
                "selection": "cosine base-only window with s closest to oracle_s",
                "fit_points": selected_info["fit_points"],
            }
            all_summary_rows.extend(
                run_staged_pipeline(
                    curves,
                    baseline,
                    selected_stage1,
                    f"{oracle_label}_matched_window_{tag(float(selected['end_fraction']))}",
                    selected_source,
                )
            )

            direct_stage1 = make_direct_oracle_stage1(selected_stage1, oracle_params)
            direct_source = {
                **selected_source,
                "stage1_s": direct_stage1["s"],
                "abs_s_error": 0.0,
                "selection": "use oracle_s directly, while keeping c1/c2 from closest cosine window",
            }
            all_summary_rows.extend(
                run_staged_pipeline(
                    curves,
                    baseline,
                    direct_stage1,
                    f"{oracle_label}_direct_oracle_s",
                    direct_source,
                )
            )

            if int(decay_weight) == 20:
                oracle_init_source = {
                    "oracle_label": oracle_label,
                    "decay_weight": decay_weight,
                    "oracle_s": oracle_params["s"],
                    "selection": "initialize Stage 2 from the full weighted cosine+WSD oracle fit, then refit on cosine with only s fixed",
                    "stage2_fixed_parameters": ["s"],
                }
                all_summary_rows.extend(
                    run_oracle_initialized_pipeline(
                        curves,
                        oracle_params,
                        f"{oracle_label}_oracle_init_fixed_s",
                        {"s"},
                        oracle_init_source,
                    )
                )

    summary = pd.DataFrame(all_summary_rows)
    summary.to_csv(OUTPUT_ROOT / f"{METHOD}_summary.csv", index=False)
    if stage1_scan_frames:
        pd.concat(stage1_scan_frames, ignore_index=True).to_csv(OUTPUT_ROOT / f"{METHOD}_stage1_window_scan.csv", index=False)
    pd.DataFrame(oracle_rows).to_csv(OUTPUT_ROOT / f"{METHOD}_oracle_summary.csv", index=False)

    best = summary[summary["stage"] == "final"].sort_values(["wsd_post_2048_rmse", "811_post_2048_rmse"]).iloc[0].to_dict()
    save_json(
        OUTPUT_ROOT / f"{METHOD}_summary.json",
        {
            "method": METHOD,
            "elapsed_seconds": timer.elapsed,
            "oracle_decay_weights": ORACLE_DECAY_WEIGHTS,
            "stage1_end_fractions": STAGE1_END_FRACTIONS,
            "stage3_starts": STAGE3_STARTS,
            "fit_stride": FIT_STRIDE,
            "baseline_source": BASELINE_SOURCE,
            "best_final": best,
            "summary_csv": str(OUTPUT_ROOT / f"{METHOD}_summary.csv"),
            "stage1_window_scan_csv": str(OUTPUT_ROOT / f"{METHOD}_stage1_window_scan.csv"),
            "oracle_summary_csv": str(OUTPUT_ROOT / f"{METHOD}_oracle_summary.csv"),
        },
    )
    print(summary.sort_values(["stage", "wsd_post_2048_rmse", "811_post_2048_rmse"]).to_string(index=False))
    print("Best final:")
    print(pd.Series(best).to_string())


if __name__ == "__main__":
    run_pipeline()
