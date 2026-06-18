from __future__ import annotations

import copy
import math

import numpy as np
import torch

from scaling_fit_utils import (
    HUBER_DELTA,
    OUTPUT_ROOT,
    PRIMARY_EVAL_START,
    SCHEDULES,
    Timer,
    load_curves,
    method_name_with_sw,
    parse_sw_mode,
    write_data_diagnostics,
    write_method_outputs,
)


METHOD = "functional_scaling_law"
FIT_START_STEP = PRIMARY_EVAL_START
FIT_STRIDE = 50
ADAM_STEPS = 2500
LBFGS_STEPS = 80
M_WIDTH = 128.0
NOISE_POINTS = 384
E_POINTS = 161
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32


def inv_softplus(x: float) -> float:
    if x > 20:
        return x
    return math.log(math.expm1(max(x, 1e-8)))


def logit(x: float) -> float:
    x = min(max(x, 1e-6), 1.0 - 1e-6)
    return math.log(x / (1.0 - x))


def encode_initial(coefficients: list[float], beta: float, s: float, sigma: float, s_w_prime: float, fit_sw: bool) -> torch.Tensor:
    raw = [inv_softplus(x) for x in coefficients]
    raw.append(logit((beta - 1.0) / 5.0))
    raw.append(logit(s / 2.0))
    raw.append(inv_softplus(sigma))
    if fit_sw:
        raw.append(inv_softplus(s_w_prime))
    return torch.tensor(raw, dtype=DTYPE, device=DEVICE)


class FSLModel(torch.nn.Module):
    def __init__(self, coefficients: list[float], beta: float, s: float, sigma: float, s_w_prime: float, fit_sw: bool):
        super().__init__()
        self.fit_sw = fit_sw
        self.raw = torch.nn.Parameter(encode_initial(coefficients, beta, s, sigma, s_w_prime, fit_sw))

    def unpack(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        coefficients = torch.nn.functional.softplus(self.raw[:4]) + 1e-10
        beta = 1.0 + 5.0 * torch.sigmoid(self.raw[4])
        s = 2.0 * torch.sigmoid(self.raw[5]) + 1e-8
        sigma = torch.nn.functional.softplus(self.raw[6]) + 1e-10
        if self.fit_sw:
            s_w_prime = torch.nn.functional.softplus(self.raw[7])
        else:
            s_w_prime = torch.zeros((), dtype=DTYPE, device=DEVICE)
        return coefficients, beta, s, sigma, s_w_prime

    def params_dict(self) -> dict[str, float]:
        coefficients, beta, s, sigma, s_w_prime = self.unpack()
        coeffs = coefficients.detach().cpu().numpy()
        return {
            "c1_constant_model_term": float(coeffs[0]),
            "c2_signal_term": float(coeffs[1]),
            "c3_minibatch_noise_term": float(coeffs[2]),
            "c4_label_noise_term": float(coeffs[3]),
            "M": M_WIDTH,
            "beta": float(beta.detach().cpu()),
            "s": float(s.detach().cpu()),
            "sigma": float(sigma.detach().cpu()),
            "s_w_prime": float(s_w_prime.detach().cpu()),
            "fit_s_w_prime": bool(self.fit_sw),
        }


class FSLCache:
    def __init__(self, curve, target_steps: np.ndarray):
        self.steps_np = target_steps.astype(np.float32)
        self.steps = torch.tensor(self.steps_np, dtype=DTYPE, device=DEVICE)
        self.loss = torch.tensor(curve.losses_at(target_steps), dtype=DTYPE, device=DEVICE)
        self.lr = torch.tensor(curve.full_lr.astype(np.float32), dtype=DTYPE, device=DEVICE)
        self.cumsum_lr = torch.cumsum(self.lr, dim=0)
        self.peak_lr = torch.tensor(float(curve.peak_lr), dtype=DTYPE, device=DEVICE)

    def intrinsic_time(self, r: torch.Tensor) -> torch.Tensor:
        r = r.to(device=DEVICE, dtype=DTYPE)
        floor_r = torch.floor(r).long()
        frac = r - floor_r.to(DTYPE)
        max_idx = len(self.lr) - 1

        idx_complete = torch.clamp(floor_r - 1, min=0, max=max_idx)
        complete = torch.where(
            floor_r > 0,
            self.cumsum_lr[idx_complete],
            torch.zeros_like(r, dtype=DTYPE, device=DEVICE),
        )
        idx_partial = torch.clamp(floor_r, min=0, max=max_idx)
        valid = floor_r <= max_idx
        partial = torch.where(valid, self.lr[idx_partial] * frac, torch.zeros_like(r, dtype=DTYPE, device=DEVICE))
        return complete + partial


def simpson_e(t: torch.Tensor, beta: torch.Tensor, s: torch.Tensor, n_points: int = E_POINTS) -> torch.Tensor:
    if n_points % 2 == 0:
        n_points += 1
    base = torch.linspace(0.0, 1.0, n_points, dtype=DTYPE, device=DEVICE)
    lower = torch.pow(torch.tensor(M_WIDTH, dtype=DTYPE, device=DEVICE), -beta)
    u = lower + (1.0 - lower) * base
    du = (1.0 - lower) / (n_points - 1)
    weights = torch.ones(n_points, dtype=DTYPE, device=DEVICE)
    weights[1:-1:2] = 4.0
    weights[2:-1:2] = 2.0
    log_term = (s - 1.0) * torch.log(u[:, None].clamp_min(1e-12)) - 2.0 * u[:, None] * t[None, :]
    values = torch.exp(log_term)
    return (weights[:, None] * values).sum(dim=0) * du / 3.0


def forgetting_kernel(t: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    return (1.0 + t.clamp_min(0.0)).pow(-2.0 + 1.0 / beta)


def expected_terms(cache: FSLCache, beta: torch.Tensor, s: torch.Tensor, sigma: torch.Tensor, s_w_prime: torch.Tensor) -> torch.Tensor:
    if len(cache.steps) == 0:
        raise ValueError("No target steps for FSL prediction.")

    max_t = float(torch.max(cache.steps).detach().cpu())
    r = torch.linspace(0.0, max_t, NOISE_POINTS, dtype=DTYPE, device=DEVICE)
    dr = r[1] - r[0]

    origin_intrinsic = cache.peak_lr * s_w_prime
    t_intrinsic = cache.intrinsic_time(cache.steps) + origin_intrinsic
    r_intrinsic = cache.intrinsic_time(r) + origin_intrinsic
    diff = t_intrinsic[:, None] - r_intrinsic[None, :]
    valid = diff > 0
    k_values = torch.where(valid, forgetting_kernel(diff, beta), torch.zeros_like(diff))

    e_values = simpson_e(r_intrinsic, beta, s)
    idx_r = torch.clamp(torch.floor(r).long(), max=len(cache.lr) - 1)
    eta_sq = cache.lr[idx_r].pow(2)

    integrand_mini = k_values * e_values[None, :] * eta_sq[None, :]
    noise_mini = (integrand_mini[:, 1:] + integrand_mini[:, :-1]).sum(dim=1) * dr / 2.0

    integrand_label = k_values * sigma.pow(2) * eta_sq[None, :]
    noise_label = (integrand_label[:, 1:] + integrand_label[:, :-1]).sum(dim=1) * dr / 2.0

    term1_value = torch.pow(torch.tensor(M_WIDTH, dtype=DTYPE, device=DEVICE), -s * beta)
    term1 = torch.ones_like(cache.steps) * term1_value
    term2 = t_intrinsic.clamp_min(1e-8).pow(-s)
    return torch.stack([term1, term2, noise_mini, noise_label], dim=1)


def predict_from_cache(model: FSLModel, cache: FSLCache) -> torch.Tensor:
    coefficients, beta, s, sigma, s_w_prime = model.unpack()
    terms = expected_terms(cache, beta, s, sigma, s_w_prime)
    return terms @ coefficients


def training_loss(model: FSLModel, cache: FSLCache) -> torch.Tensor:
    pred = predict_from_cache(model, cache)
    safe_pred = pred.clamp_min(1e-8)
    residual = torch.log(cache.loss) - torch.log(safe_pred)
    abs_r = torch.abs(residual)
    huber = torch.where(abs_r <= HUBER_DELTA, 0.5 * residual * residual, HUBER_DELTA * (abs_r - 0.5 * HUBER_DELTA))
    penalty = torch.relu(1e-6 - pred).pow(2).mean() * 1e6
    return huber.mean() + penalty


def train_one(initial: tuple[list[float], float, float, float, float], cache: FSLCache, fit_sw: bool):
    coefficients, beta, s, sigma, s_w_prime = initial
    model = FSLModel(coefficients, beta, s, sigma, s_w_prime, fit_sw).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-2)

    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    history = []

    for step in range(ADAM_STEPS):
        optimizer.zero_grad()
        value = training_loss(model, cache)
        value.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        loss_value = float(value.detach().cpu())
        history.append(loss_value)
        if loss_value < best_loss:
            best_loss = loss_value
            best_state = copy.deepcopy(model.state_dict())
        if step % 500 == 0:
            print(f"FSL Adam step {step:4d}: {loss_value:.6e}")

    model.load_state_dict(best_state)
    lbfgs = torch.optim.LBFGS(model.parameters(), lr=0.7, max_iter=LBFGS_STEPS, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        lbfgs.zero_grad()
        value = training_loss(model, cache)
        value.backward()
        return value

    lbfgs.step(closure)
    final_loss = float(training_loss(model, cache).detach().cpu())
    if final_loss < best_loss:
        best_loss = final_loss
        best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return best_loss, model.params_dict(), {"initial": initial, "adam_history": history}


def model_from_params(params: dict[str, float]) -> FSLModel:
    coefficients = [
        params["c1_constant_model_term"],
        params["c2_signal_term"],
        params["c3_minibatch_noise_term"],
        params["c4_label_noise_term"],
    ]
    return FSLModel(coefficients, params["beta"], params["s"], params["sigma"], params.get("s_w_prime", 0.0), fit_sw=True).to(DEVICE)


@torch.no_grad()
def predict_curve(params: dict[str, float], curve) -> np.ndarray:
    model = model_from_params(params)
    cache = FSLCache(curve, curve.full_steps)
    pred = predict_from_cache(model, cache)
    return pred.detach().cpu().numpy().astype(np.float64)


def fit_model(fit_sw: bool):
    curves, diagnostics = load_curves()
    write_data_diagnostics(diagnostics)
    train = curves["cosine"]
    fit_steps = train.fit_steps(FIT_START_STEP, FIT_STRIDE)
    cache = FSLCache(train, fit_steps)

    base_initial_sets = [
        ([5000.0, 2.0, 1000.0, 1000.0], 2.5, 0.6, 1.0),
        ([1000.0, 5.0, 3000.0, 500.0], 3.0, 0.5, 0.5),
        ([8000.0, 1.0, 100.0, 5000.0], 2.0, 0.8, 1.5),
    ]
    sw_initials = [0.0, 2048.0, 8192.0]
    if fit_sw:
        initial_sets = [
            (coeffs, beta, s, sigma, sw)
            for coeffs, beta, s, sigma in base_initial_sets
            for sw in sw_initials
        ]
    else:
        initial_sets = [(coeffs, beta, s, sigma, 0.0) for coeffs, beta, s, sigma in base_initial_sets]

    best_loss = float("inf")
    best_params = None
    histories = []
    for initial in initial_sets:
        loss_value, params, history = train_one(initial, cache, fit_sw)
        history["best_loss"] = loss_value
        history["best_params"] = params
        histories.append(history)
        if loss_value < best_loss:
            best_loss = loss_value
            best_params = params
        print(f"FSL init done: {loss_value:.6e} {params}")

    if best_params is None:
        raise RuntimeError("FSL fitting failed.")

    best_params["S_W_cosine"] = float(curves["cosine"].peak_lr * best_params["s_w_prime"])

    predictions = {name: predict_curve(best_params, curves[name]) for name in SCHEDULES}
    fit_info = {
        "method": METHOD,
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
        "best_training_huber_mean": best_loss,
        "histories": histories,
        "sw_mode": "fit" if fit_sw else "fixed",
        "peak_lrs": {name: curves[name].peak_lr for name in SCHEDULES},
        "implementation_note": "This follows the old notebook expected_R route, not the practical LLM FSL ansatz in the paper.",
    }
    return curves, best_params, predictions, fit_info


def main() -> None:
    fit_sw = parse_sw_mode()
    output_method = method_name_with_sw(METHOD, fit_sw)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with Timer() as timer:
        curves, params, predictions, fit_info = fit_model(fit_sw)
    fit_info["elapsed_seconds"] = timer.elapsed
    fit_info["base_method"] = METHOD
    fit_info["method"] = output_method
    write_method_outputs(output_method, params, curves, predictions, fit_info)
    print(f"{output_method} done in {timer.elapsed:.2f}s")
    print(params)


if __name__ == "__main__":
    main()
