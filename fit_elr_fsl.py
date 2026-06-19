import argparse
import copy
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scaling_fit_utils import (
    HUBER_DELTA,
    OUTPUT_ROOT,
    PRIMARY_EVAL_START,
    SCHEDULES,
    Timer,
    load_curves,
    write_data_diagnostics,
    write_method_outputs,
)


METHOD_ZERO_LAMBDA = "elr_functional_scaling_law_lambda_zero"
METHOD_FREE_LAMBDA = "elr_functional_scaling_law_free_lambda"
METHOD_FIXED_PREFIX = "elr_functional_scaling_law_lambda_fixed"

FIT_START_STEP = PRIMARY_EVAL_START
FIT_STRIDE = 50
ADAM_STEPS = 2200
LBFGS_STEPS = 70

M_WIDTH = 128.0
NOISE_POINTS = 384
E_POINTS = 161
LAMBDA_MAX = 2.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda-mode", choices=["fit", "fixed"], default="fixed")
    parser.add_argument("--lambda-value", type=float, default=0.0)
    return parser.parse_args()


def inv_softplus(x: float) -> float:
    if x > 20.0:
        return x
    return math.log(math.expm1(max(x, 1.0e-8)))


def logit(x: float) -> float:
    x = min(max(x, 1.0e-6), 1.0 - 1.0e-6)
    return math.log(x / (1.0 - x))


def encode_initial(initial: dict[str, float]) -> torch.Tensor:
    raw = [inv_softplus(initial[name]) for name in ["c1", "c2", "c3", "c4"]]
    raw.append(logit((initial["beta"] - 1.0) / 5.0))
    raw.append(logit(initial["s"] / 2.0))
    raw.append(inv_softplus(initial["sigma"]))
    raw.append(logit(initial["lambda_wd"] / LAMBDA_MAX))
    raw.append(inv_softplus(initial["C_u"]))
    raw.append(inv_softplus(initial["n0"]))
    return torch.tensor(raw, dtype=DTYPE, device=DEVICE)


class ELRFSLModel(torch.nn.Module):
    def __init__(self, initial: dict[str, float], fit_lambda: bool, fixed_lambda: float = 0.0):
        super().__init__()
        self.fit_lambda = fit_lambda
        self.fixed_lambda = float(fixed_lambda)
        self.raw = torch.nn.Parameter(encode_initial(initial))

    def unpack(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        coeffs = torch.nn.functional.softplus(self.raw[:4]) + 1.0e-10
        beta = 1.0 + 5.0 * torch.sigmoid(self.raw[4])
        s = 2.0 * torch.sigmoid(self.raw[5]) + 1.0e-8
        sigma = torch.nn.functional.softplus(self.raw[6]) + 1.0e-10
        if self.fit_lambda:
            lambda_wd = LAMBDA_MAX * torch.sigmoid(self.raw[7])
        else:
            lambda_wd = torch.tensor(self.fixed_lambda, dtype=DTYPE, device=DEVICE)
        C_u = torch.nn.functional.softplus(self.raw[8]) + 1.0e-12
        n0 = torch.nn.functional.softplus(self.raw[9]) + 1.0e-12
        return coeffs, beta, s, sigma, lambda_wd, C_u, n0

    def params_dict(self) -> dict[str, float | bool | str]:
        coeffs, beta, s, sigma, lambda_wd, C_u, n0 = self.unpack()
        coeff_values = coeffs.detach().cpu().numpy()
        return {
            "c1_constant_model_term": float(coeff_values[0]),
            "c2_signal_term": float(coeff_values[1]),
            "c3_minibatch_noise_term": float(coeff_values[2]),
            "c4_label_noise_term": float(coeff_values[3]),
            "M": M_WIDTH,
            "beta": float(beta.detach().cpu()),
            "s": float(s.detach().cpu()),
            "sigma": float(sigma.detach().cpu()),
            "lambda_wd": float(lambda_wd.detach().cpu()),
            "C_u": float(C_u.detach().cpu()),
            "n0": float(n0.detach().cpu()),
            "fit_lambda": bool(self.fit_lambda),
            "fixed_lambda": None if self.fit_lambda else self.fixed_lambda,
            "lambda_max": LAMBDA_MAX,
            "intrinsic_time": "sum eta_k / sqrt(n_k)",
            "norm_recurrence": "n_{k+1}=(1-eta_k*lambda)^2*n_k+eta_k^2*C_u",
        }


class ELRFSLCache:
    def __init__(self, curve, target_steps: np.ndarray):
        self.steps_np = target_steps.astype(np.float32)
        self.steps = torch.tensor(self.steps_np, dtype=DTYPE, device=DEVICE)
        self.loss = torch.tensor(curve.losses_at(target_steps), dtype=DTYPE, device=DEVICE)
        self.lr = torch.tensor(curve.full_lr.astype(np.float32), dtype=DTYPE, device=DEVICE)

    def norm_and_elr(
        self,
        lambda_wd: torch.Tensor,
        C_u: torch.Tensor,
        n0: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        eta = self.lr
        a = (1.0 - eta * lambda_wd).pow(2).clamp_min(1.0e-12)
        b = eta.pow(2) * C_u

        log_p_next = torch.cumsum(torch.log(a), dim=0)
        p_next = torch.exp(log_p_next).clamp_min(1.0e-30)
        p_current = torch.cat([torch.ones(1, dtype=DTYPE, device=DEVICE), p_next[:-1]])

        increments = b / p_next
        sum_before = torch.cat(
            [
                torch.zeros(1, dtype=DTYPE, device=DEVICE),
                torch.cumsum(increments, dim=0)[:-1],
            ]
        )
        norm_square = (p_current * (n0 + sum_before)).clamp_min(1.0e-12)
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


def expected_terms(
    cache: ELRFSLCache,
    beta: torch.Tensor,
    s: torch.Tensor,
    sigma: torch.Tensor,
    lambda_wd: torch.Tensor,
    C_u: torch.Tensor,
    n0: torch.Tensor,
) -> torch.Tensor:
    if len(cache.steps) == 0:
        raise ValueError("No target steps for ELR-FSL prediction.")

    _, _, elr = cache.norm_and_elr(lambda_wd, C_u, n0)
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


def predict_from_cache(model: ELRFSLModel, cache: ELRFSLCache) -> torch.Tensor:
    coeffs, beta, s, sigma, lambda_wd, C_u, n0 = model.unpack()
    terms = expected_terms(cache, beta, s, sigma, lambda_wd, C_u, n0)
    return terms @ coeffs


def log_huber_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    safe_pred = pred.clamp_min(1.0e-8)
    residual = torch.log(target) - torch.log(safe_pred)
    abs_r = torch.abs(residual)
    huber = torch.where(
        abs_r <= HUBER_DELTA,
        0.5 * residual * residual,
        HUBER_DELTA * (abs_r - 0.5 * HUBER_DELTA),
    )
    penalty = torch.relu(1.0e-6 - pred).pow(2).mean() * 1.0e6
    return huber.mean() + penalty


def training_loss(model: ELRFSLModel, cache: ELRFSLCache) -> torch.Tensor:
    pred = predict_from_cache(model, cache)
    return log_huber_loss(pred, cache.loss)


def train_one(initial: dict[str, float], cache: ELRFSLCache, fit_lambda: bool, fixed_lambda: float):
    model = ELRFSLModel(initial, fit_lambda, fixed_lambda).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=2.0e-2)

    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    history = []

    for step in range(ADAM_STEPS):
        optimizer.zero_grad(set_to_none=True)
        value = training_loss(model, cache)
        value.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()

        loss_value = float(value.detach().cpu())
        history.append(loss_value)
        if math.isfinite(loss_value) and loss_value < best_loss:
            best_loss = loss_value
            best_state = copy.deepcopy(model.state_dict())
        if step % 500 == 0:
            label = "free-lambda" if fit_lambda else f"lambda-fixed-{fixed_lambda:g}"
            print(f"ELR-FSL {label} Adam step {step:4d}: {loss_value:.6e}")

    model.load_state_dict(best_state)
    lbfgs = torch.optim.LBFGS(model.parameters(), lr=0.7, max_iter=LBFGS_STEPS, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        lbfgs.zero_grad(set_to_none=True)
        value = training_loss(model, cache)
        value.backward()
        return value

    lbfgs.step(closure)
    final_loss = float(training_loss(model, cache).detach().cpu())
    if math.isfinite(final_loss) and final_loss < best_loss:
        best_loss = final_loss
        best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return best_loss, model.params_dict(), {"initial": initial, "adam_history": history}, copy.deepcopy(model.state_dict())


def initial_sets() -> list[dict[str, float]]:
    return [
        {
            "c1": 7999.9,
            "c2": 1.2,
            "c3": 95.0,
            "c4": 5000.0,
            "beta": 2.03,
            "s": 0.82,
            "sigma": 0.17,
            "lambda_wd": 1.0e-4,
            "C_u": 1.0e-4,
            "n0": 1.0,
        },
        {
            "c1": 5000.0,
            "c2": 2.0,
            "c3": 1000.0,
            "c4": 1000.0,
            "beta": 2.5,
            "s": 0.6,
            "sigma": 1.0,
            "lambda_wd": 1.0e-2,
            "C_u": 1.0,
            "n0": 1.0,
        },
        {
            "c1": 1000.0,
            "c2": 5.0,
            "c3": 3000.0,
            "c4": 500.0,
            "beta": 3.0,
            "s": 0.5,
            "sigma": 0.5,
            "lambda_wd": 5.0e-2,
            "C_u": 10.0,
            "n0": 1.0,
        },
        {
            "c1": 8000.0,
            "c2": 1.0,
            "c3": 100.0,
            "c4": 5000.0,
            "beta": 2.0,
            "s": 0.8,
            "sigma": 1.5,
            "lambda_wd": 0.2,
            "C_u": 0.1,
            "n0": 0.3,
        },
    ]


def load_model_state(
    initial: dict[str, float],
    state: dict[str, torch.Tensor],
    fit_lambda: bool,
    fixed_lambda: float,
) -> ELRFSLModel:
    model = ELRFSLModel(initial, fit_lambda, fixed_lambda).to(DEVICE)
    model.load_state_dict(state)
    return model


def model_from_params(params: dict[str, float | bool]) -> ELRFSLModel:
    initial = {
        "c1": float(params["c1_constant_model_term"]),
        "c2": float(params["c2_signal_term"]),
        "c3": float(params["c3_minibatch_noise_term"]),
        "c4": float(params["c4_label_noise_term"]),
        "beta": float(params["beta"]),
        "s": float(params["s"]),
        "sigma": float(params["sigma"]),
        "lambda_wd": max(float(params["lambda_wd"]), 1.0e-8),
        "C_u": float(params["C_u"]),
        "n0": float(params["n0"]),
    }
    fixed_lambda = 0.0 if bool(params["fit_lambda"]) else float(params["lambda_wd"])
    return ELRFSLModel(initial, bool(params["fit_lambda"]), fixed_lambda).to(DEVICE)


@torch.no_grad()
def predict_curve(model: ELRFSLModel, curve) -> np.ndarray:
    cache = ELRFSLCache(curve, curve.full_steps)
    pred = predict_from_cache(model, cache)
    return pred.detach().cpu().numpy().astype(np.float64)


@torch.no_grad()
def elr_dynamics(model: ELRFSLModel, curve) -> pd.DataFrame:
    cache = ELRFSLCache(curve, curve.full_steps)
    _, _, _, _, lambda_wd, C_u, n0 = model.unpack()
    norm_square, norm, elr = cache.norm_and_elr(lambda_wd, C_u, n0)
    return pd.DataFrame(
        {
            "step": curve.full_steps,
            "lr": curve.full_lr,
            "norm_square": norm_square.detach().cpu().numpy().astype(np.float64),
            "norm": norm.detach().cpu().numpy().astype(np.float64),
            "elr": elr.detach().cpu().numpy().astype(np.float64),
        }
    )


def save_elr_diagnostics(method: str, model: ELRFSLModel, curves) -> None:
    method_dir = OUTPUT_ROOT / method
    for name in SCHEDULES:
        frame = elr_dynamics(model, curves[name])
        frame.to_csv(method_dir / f"{name}_elr_dynamics.csv", index=False)


def fit_model(fit_lambda: bool, fixed_lambda: float):
    curves, diagnostics = load_curves()
    write_data_diagnostics(diagnostics)
    train = curves["cosine"]
    fit_steps = train.fit_steps(FIT_START_STEP, FIT_STRIDE)
    cache = ELRFSLCache(train, fit_steps)

    best_loss = float("inf")
    best_params = None
    best_state = None
    best_initial = None
    histories = []

    for initial in initial_sets():
        loss_value, params, history, state = train_one(initial, cache, fit_lambda, fixed_lambda)
        history["best_loss"] = loss_value
        history["best_params"] = params
        histories.append(history)
        print("ELR-FSL init done:", f"{loss_value:.6e}", params)
        if loss_value < best_loss:
            best_loss = loss_value
            best_params = params
            best_state = state
            best_initial = initial

    if best_params is None or best_state is None or best_initial is None:
        raise RuntimeError("ELR-FSL fitting failed.")

    best_model = load_model_state(best_initial, best_state, fit_lambda, fixed_lambda)
    predictions = {name: predict_curve(best_model, curves[name]) for name in SCHEDULES}

    fit_info = {
        "method": method_name(fit_lambda, fixed_lambda),
        "optimizer": "Adam followed by torch LBFGS",
        "device": str(DEVICE),
        "dtype": str(DTYPE),
        "fit_schedule": "cosine",
        "fit_start_step": FIT_START_STEP,
        "fit_stride": FIT_STRIDE,
        "fit_points": int(len(fit_steps)),
        "huber_delta": HUBER_DELTA,
        "adam_steps": ADAM_STEPS,
        "lbfgs_steps": LBFGS_STEPS,
        "M_width": M_WIDTH,
        "noise_points": NOISE_POINTS,
        "e_points": E_POINTS,
        "lambda_mode": "fit" if fit_lambda else "fixed",
        "fixed_lambda": None if fit_lambda else fixed_lambda,
        "lambda_max": LAMBDA_MAX,
        "best_training_huber_mean": best_loss,
        "histories": histories,
        "implementation_note": "FSL expected_R route with every LR occurrence replaced by ELR=eta/sqrt(n), where n follows the discrete norm-square recurrence.",
    }
    return curves, best_params, predictions, fit_info, best_model


def format_lambda_for_name(value: float) -> str:
    text = f"{value:g}".replace("-", "m").replace(".", "p")
    return text.replace("+", "")


def method_name(fit_lambda: bool, fixed_lambda: float) -> str:
    if fit_lambda:
        return METHOD_FREE_LAMBDA
    if abs(fixed_lambda) < 1.0e-15:
        return METHOD_ZERO_LAMBDA
    return f"{METHOD_FIXED_PREFIX}_{format_lambda_for_name(fixed_lambda)}"


def run_one(fit_lambda: bool, fixed_lambda: float) -> None:
    method = method_name(fit_lambda, fixed_lambda)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with Timer() as timer:
        curves, params, predictions, fit_info, model = fit_model(fit_lambda, fixed_lambda)
    fit_info["elapsed_seconds"] = timer.elapsed
    write_method_outputs(method, params, curves, predictions, fit_info)
    save_elr_diagnostics(method, model, curves)
    print(f"{method} done in {timer.elapsed:.2f}s")
    print(params)


def main() -> None:
    args = parse_args()
    if args.lambda_value < 0:
        raise ValueError("--lambda-value must be nonnegative.")
    run_one(fit_lambda=args.lambda_mode == "fit", fixed_lambda=args.lambda_value)


if __name__ == "__main__":
    main()
