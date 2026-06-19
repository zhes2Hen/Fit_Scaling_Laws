from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fit_multi_power_law import (  # noqa: E402
    DEVICE,
    DTYPE,
    FIT_START_STEP,
    MPLCache,
    inv_softplus,
    logit,
)
from scaling_fit_utils import HUBER_DELTA  # noqa: E402


ALL_PARAMETERS = ("L0", "A", "alpha", "B", "C", "beta", "gamma", "s_w_prime")


def _raw_from_value(name: str, value: float) -> float:
    if name == "L0":
        return logit((value - 1.0) / 4.0)
    if name in {"A", "B", "C", "s_w_prime"}:
        return inv_softplus(value)
    if name in {"alpha", "beta", "gamma"}:
        return logit(value / 2.0)
    raise KeyError(name)


def _value_from_raw(name: str, raw: torch.Tensor) -> torch.Tensor:
    if name == "L0":
        return 1.0 + 4.0 * torch.sigmoid(raw)
    if name in {"A", "B", "C"}:
        return torch.nn.functional.softplus(raw) + 1e-10
    if name == "s_w_prime":
        return torch.nn.functional.softplus(raw)
    if name in {"alpha", "beta", "gamma"}:
        return 2.0 * torch.sigmoid(raw) + 1e-8
    raise KeyError(name)


def _log_huber(observed: torch.Tensor, predicted: torch.Tensor) -> torch.Tensor:
    residual = torch.log(observed) - torch.log(predicted.clamp_min(1e-8))
    absolute = residual.abs()
    return torch.where(
        absolute <= HUBER_DELTA,
        0.5 * residual.square(),
        HUBER_DELTA * (absolute - 0.5 * HUBER_DELTA),
    ).mean()


class BasePowerModel(torch.nn.Module):
    def __init__(self, initial: dict[str, float]):
        super().__init__()
        self.raw = torch.nn.Parameter(
            torch.tensor(
                [
                    logit((initial["L0"] - 1.0) / 4.0),
                    inv_softplus(initial["A"]),
                    logit(initial["alpha"] / 2.0),
                ],
                dtype=DTYPE,
                device=DEVICE,
            )
        )

    def unpack(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        l0 = 1.0 + 4.0 * torch.sigmoid(self.raw[0])
        a = torch.nn.functional.softplus(self.raw[1]) + 1e-10
        alpha = 2.0 * torch.sigmoid(self.raw[2]) + 1e-8
        return l0, a, alpha

    def params_dict(self) -> dict[str, float]:
        l0, a, alpha = self.unpack()
        return {
            "L0": float(l0.detach().cpu()),
            "A": float(a.detach().cpu()),
            "alpha": float(alpha.detach().cpu()),
        }


def _base_prediction(model: BasePowerModel, cache: MPLCache) -> torch.Tensor:
    l0, a, alpha = model.unpack()
    return l0 + a * cache.s1.clamp_min(1e-12).pow(-alpha)


def fit_base_only(
    curve,
    steps: np.ndarray,
    initial: dict[str, float],
    adam_steps: int = 1800,
    lbfgs_steps: int = 100,
) -> tuple[dict[str, float], dict[str, object]]:
    cache = MPLCache(curve, steps)
    model = BasePowerModel(initial).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-2)

    best_value = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    history: list[dict[str, float | int]] = []
    for step in range(adam_steps):
        optimizer.zero_grad()
        value = _log_huber(cache.loss, _base_prediction(model, cache))
        value.backward()
        optimizer.step()
        scalar = float(value.detach().cpu())
        if step % 25 == 0 or step == adam_steps - 1:
            history.append({"step": step, "loss": scalar})
        if scalar < best_value:
            best_value = scalar
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    if lbfgs_steps > 0:
        lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=0.7,
            max_iter=lbfgs_steps,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            lbfgs.zero_grad()
            value = _log_huber(cache.loss, _base_prediction(model, cache))
            value.backward()
            return value

        lbfgs.step(closure)
        final_value = float(
            _log_huber(cache.loss, _base_prediction(model, cache)).detach().cpu()
        )
        if final_value < best_value:
            best_value = final_value
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model.params_dict(), {
        "best_loss": best_value,
        "fit_points": int(len(steps)),
        "fit_step_min": int(steps.min()),
        "fit_step_max": int(steps.max()),
        "history": history,
    }


class FrozenParameterMPL(torch.nn.Module):
    def __init__(
        self,
        initial: dict[str, float],
        fixed_parameters: set[str],
    ):
        super().__init__()
        self.fixed_parameters = set(fixed_parameters)
        self.trainable_names = [
            name for name in ALL_PARAMETERS if name not in self.fixed_parameters
        ]
        if not self.trainable_names:
            raise ValueError("At least one MPL parameter must remain trainable.")

        for name in self.fixed_parameters:
            self.register_buffer(
                f"fixed_{name}",
                torch.tensor(float(initial[name]), dtype=DTYPE, device=DEVICE),
            )

        raw = torch.tensor(
            [_raw_from_value(name, float(initial[name])) for name in self.trainable_names],
            dtype=DTYPE,
            device=DEVICE,
        )
        self.raw = torch.nn.Parameter(raw.clone())
        self.register_buffer("raw_reference", raw.clone())

    def value(self, name: str) -> torch.Tensor:
        if name in self.fixed_parameters:
            return getattr(self, f"fixed_{name}")
        return _value_from_raw(name, self.raw[self.trainable_names.index(name)])

    def unpack(self) -> tuple[torch.Tensor, ...]:
        return tuple(self.value(name) for name in ALL_PARAMETERS)

    def params_dict(self) -> dict[str, float | bool]:
        result: dict[str, float | bool] = {
            name: float(value.detach().cpu())
            for name, value in zip(ALL_PARAMETERS, self.unpack())
        }
        result["fit_s_w_prime"] = "s_w_prime" not in self.fixed_parameters
        return result


def _predict(model: FrozenParameterMPL, cache: MPLCache) -> torch.Tensor:
    l0, a, alpha, b, c, beta, gamma, s_w_prime = model.unpack()
    x = c * cache.span * cache.active_lr[None, :].clamp_min(1e-12).pow(-gamma)
    kernel = 1.0 - (1.0 + x).pow(-beta)
    loss_drop = (cache.drop_mask * kernel).sum(dim=1)
    shifted_s1 = cache.s1 + cache.peak_lr * s_w_prime
    return l0 + a * shifted_s1.clamp_min(1e-12).pow(-alpha) - b * loss_drop


def selected_steps(curve, start_fraction: float, stride: int) -> np.ndarray:
    if not 0.0 <= start_fraction < 1.0:
        raise ValueError("start_fraction must be in [0, 1).")
    if stride <= 0:
        raise ValueError("stride must be positive.")
    boundary = int(math.floor(start_fraction * int(curve.full_steps[-1])))
    start = max(FIT_START_STEP, boundary)
    steps = curve.observed_steps[curve.observed_steps >= start]
    steps = steps[(steps - start) % stride == 0]
    if len(steps) == 0 or steps[-1] != curve.observed_steps[-1]:
        steps = np.append(steps, curve.observed_steps[-1])
    return np.unique(steps.astype(np.int64))


def fit_with_frozen_parameters(
    curve,
    initial: dict[str, float],
    fixed_parameters: set[str],
    start_fraction: float,
    stride: int,
    adam_steps: int = 2500,
    lbfgs_steps: int = 120,
    parameter_regularization: float = 0.0,
) -> tuple[dict[str, float | bool], dict[str, object]]:
    steps = selected_steps(curve, start_fraction, stride)
    cache = MPLCache(curve, steps)
    model = FrozenParameterMPL(initial, fixed_parameters).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-2)

    best_value = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    history: list[dict[str, float | int]] = []

    def objective() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prediction = _predict(model, cache)
        huber = _log_huber(cache.loss, prediction)
        regularizer = float(parameter_regularization) * (
            model.raw - model.raw_reference
        ).square().mean()
        positivity = torch.relu(1e-6 - prediction).square().mean() * 1e6
        return huber + regularizer + positivity, huber, regularizer

    for step in range(adam_steps):
        optimizer.zero_grad()
        total, huber, regularizer = objective()
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        scalar = float(total.detach().cpu())
        if step % 25 == 0 or step == adam_steps - 1:
            history.append(
                {
                    "step": step,
                    "total": scalar,
                    "huber": float(huber.detach().cpu()),
                    "parameter_regularizer": float(regularizer.detach().cpu()),
                }
            )
        if scalar < best_value:
            best_value = scalar
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    if lbfgs_steps > 0:
        lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=0.7,
            max_iter=lbfgs_steps,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            lbfgs.zero_grad()
            total, _, _ = objective()
            total.backward()
            return total

        lbfgs.step(closure)
        final_total, _, _ = objective()
        final_value = float(final_total.detach().cpu())
        if final_value < best_value:
            best_value = final_value
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    total, huber, regularizer = objective()
    return model.params_dict(), {
        "best_total_loss": best_value,
        "final_total_loss": float(total.detach().cpu()),
        "final_huber": float(huber.detach().cpu()),
        "final_parameter_regularizer": float(regularizer.detach().cpu()),
        "fit_points": int(len(steps)),
        "fit_step_min": int(steps.min()),
        "fit_step_max": int(steps.max()),
        "fixed_parameters": sorted(fixed_parameters),
        "trainable_parameters": [
            name for name in ALL_PARAMETERS if name not in fixed_parameters
        ],
        "history": history,
    }
