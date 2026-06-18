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
    huber_np,
    load_curves,
    method_name_with_sw,
    parse_sw_mode,
    write_data_diagnostics,
    write_method_outputs,
)


METHOD = "multi_power_law"
FIT_START_STEP = PRIMARY_EVAL_START
FIT_STRIDE = 50
ADAM_STEPS = 2500
LBFGS_STEPS = 120
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32


def inv_softplus(x: float) -> float:
    if x > 20:
        return x
    return math.log(math.expm1(max(x, 1e-8)))


def logit(x: float) -> float:
    x = min(max(x, 1e-6), 1.0 - 1e-6)
    return math.log(x / (1.0 - x))


def encode_initial(params: list[float], fit_sw: bool) -> torch.Tensor:
    l0, a, alpha, b, c, beta, gamma = params[:7]
    raw = [
        logit((l0 - 1.0) / 4.0),
        inv_softplus(a),
        logit(alpha / 2.0),
        inv_softplus(b),
        inv_softplus(c),
        logit(beta / 2.0),
        logit(gamma / 2.0),
    ]
    if fit_sw:
        raw.append(inv_softplus(params[7]))
    return torch.tensor(raw, dtype=DTYPE, device=DEVICE)


class MPLModel(torch.nn.Module):
    def __init__(self, initial_params: list[float], fit_sw: bool):
        super().__init__()
        self.fit_sw = fit_sw
        self.raw = torch.nn.Parameter(encode_initial(initial_params, fit_sw))

    def unpack(self) -> tuple[torch.Tensor, ...]:
        l0 = 1.0 + 4.0 * torch.sigmoid(self.raw[0])
        a = torch.nn.functional.softplus(self.raw[1]) + 1e-10
        alpha = 2.0 * torch.sigmoid(self.raw[2]) + 1e-8
        b = torch.nn.functional.softplus(self.raw[3]) + 1e-10
        c = torch.nn.functional.softplus(self.raw[4]) + 1e-10
        beta = 2.0 * torch.sigmoid(self.raw[5]) + 1e-8
        gamma = 2.0 * torch.sigmoid(self.raw[6]) + 1e-8
        if self.fit_sw:
            s_w_prime = torch.nn.functional.softplus(self.raw[7])
        else:
            s_w_prime = torch.zeros((), dtype=DTYPE, device=DEVICE)
        return l0, a, alpha, b, c, beta, gamma, s_w_prime

    def params_dict(self) -> dict[str, float]:
        l0, a, alpha, b, c, beta, gamma, s_w_prime = self.unpack()
        return {
            "L0": float(l0.detach().cpu()),
            "A": float(a.detach().cpu()),
            "alpha": float(alpha.detach().cpu()),
            "B": float(b.detach().cpu()),
            "C": float(c.detach().cpu()),
            "beta": float(beta.detach().cpu()),
            "gamma": float(gamma.detach().cpu()),
            "s_w_prime": float(s_w_prime.detach().cpu()),
            "fit_s_w_prime": bool(self.fit_sw),
        }


class MPLCache:
    def __init__(self, curve, target_steps: np.ndarray):
        lr_np = curve.full_lr.astype(np.float32)
        s1_np = np.cumsum(lr_np, dtype=np.float64).astype(np.float32)
        s1_before_np = np.concatenate([[0.0], s1_np[:-1]]).astype(np.float32)
        drop_np = np.zeros_like(lr_np, dtype=np.float32)
        drop_np[1:] = np.maximum(lr_np[:-1] - lr_np[1:], 0.0)
        active = np.where(drop_np > 0)[0].astype(np.int64)

        self.steps = torch.tensor(target_steps, dtype=torch.long, device=DEVICE)
        self.loss = torch.tensor(curve.losses_at(target_steps), dtype=DTYPE, device=DEVICE)
        self.s1 = torch.tensor(s1_np[target_steps], dtype=DTYPE, device=DEVICE)
        self.peak_lr = torch.tensor(float(curve.peak_lr), dtype=DTYPE, device=DEVICE)
        self.active_steps = torch.tensor(active, dtype=torch.long, device=DEVICE)
        self.active_lr = torch.tensor(lr_np[active], dtype=DTYPE, device=DEVICE)
        self.active_drop = torch.tensor(drop_np[active], dtype=DTYPE, device=DEVICE)
        active_before = torch.tensor(s1_before_np[active], dtype=DTYPE, device=DEVICE)
        target_s1 = self.s1[:, None]
        span = (target_s1 - active_before[None, :]).clamp_min(0.0)
        mask = self.active_steps[None, :] <= self.steps[:, None]
        self.span = span
        self.drop_mask = self.active_drop[None, :] * mask.to(DTYPE)


def predict_from_cache(model: MPLModel, cache: MPLCache) -> torch.Tensor:
    l0, a, alpha, b, c, beta, gamma, s_w_prime = model.unpack()
    x = c * cache.span * cache.active_lr[None, :].clamp_min(1e-12).pow(-gamma)
    g = 1.0 - (1.0 + x).pow(-beta)
    loss_drop = (cache.drop_mask * g).sum(dim=1)
    shifted_s1 = cache.s1 + cache.peak_lr * s_w_prime
    return l0 + a * shifted_s1.clamp_min(1e-12).pow(-alpha) - b * loss_drop


def training_loss(model: MPLModel, cache: MPLCache) -> torch.Tensor:
    pred = predict_from_cache(model, cache)
    safe_pred = pred.clamp_min(1e-8)
    residual = torch.log(cache.loss) - torch.log(safe_pred)
    abs_r = torch.abs(residual)
    huber = torch.where(abs_r <= HUBER_DELTA, 0.5 * residual * residual, HUBER_DELTA * (abs_r - 0.5 * HUBER_DELTA))
    penalty = torch.relu(1e-6 - pred).pow(2).mean() * 1e6
    return huber.mean() + penalty


def train_one(initial_params: list[float], cache: MPLCache, fit_sw: bool) -> tuple[float, dict[str, float], dict[str, object]]:
    model = MPLModel(initial_params, fit_sw).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-2)

    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    history = []

    for step in range(ADAM_STEPS):
        optimizer.zero_grad()
        loss = training_loss(model, cache)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        value = float(loss.detach().cpu())
        history.append(value)
        if value < best_loss:
            best_loss = value
            best_state = copy.deepcopy(model.state_dict())
        if step % 500 == 0:
            print(f"MPL Adam step {step:4d}: {value:.6e}")

    model.load_state_dict(best_state)
    lbfgs = torch.optim.LBFGS(model.parameters(), lr=0.8, max_iter=LBFGS_STEPS, line_search_fn="strong_wolfe")

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

    return best_loss, model.params_dict(), {"adam_history": history, "initial_params": initial_params}


def model_from_params(params: dict[str, float]) -> MPLModel:
    ordered = [
        params["L0"],
        params["A"],
        params["alpha"],
        params["B"],
        params["C"],
        params["beta"],
        params["gamma"],
        params.get("s_w_prime", 0.0),
    ]
    model = MPLModel(ordered, fit_sw=True).to(DEVICE)
    return model


@torch.no_grad()
def predict_curve(params: dict[str, float], curve) -> np.ndarray:
    model = model_from_params(params)
    full_steps = curve.full_steps
    chunks = []
    chunk_size = 512
    for start in range(0, len(full_steps), chunk_size):
        target_steps = full_steps[start : start + chunk_size]
        cache = MPLCache(curve, target_steps)
        chunks.append(predict_from_cache(model, cache).detach().cpu().numpy().astype(np.float64))
    return np.concatenate(chunks)


def fit_model(fit_sw: bool):
    curves, diagnostics = load_curves()
    write_data_diagnostics(diagnostics)
    train = curves["cosine"]
    fit_steps = train.fit_steps(FIT_START_STEP, FIT_STRIDE)
    cache = MPLCache(train, fit_steps)

    base_initial_sets = [
        [2.6, 0.6, 0.45, 450.0, 0.8, 0.6, 0.6],
        [2.4, 1.5, 0.55, 150.0, 1.5, 0.5, 0.8],
        [2.8, 0.3, 0.35, 900.0, 0.2, 0.9, 0.4],
    ]
    sw_initials = [0.0, 2048.0, 8192.0]
    if fit_sw:
        initial_sets = [[*base, sw] for base in base_initial_sets for sw in sw_initials]
    else:
        initial_sets = base_initial_sets

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
        print(f"MPL init done: {loss_value:.6e} {params}")

    if best_params is None:
        raise RuntimeError("MPL fitting failed.")

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
        "best_training_huber_mean": best_loss,
        "histories": histories,
        "sw_mode": "fit" if fit_sw else "fixed",
        "peak_lrs": {name: curves[name].peak_lr for name in SCHEDULES},
        "implementation_note": "Uses positive LR drops eta[k-1]-eta[k] and subtracts B*LD. The fitted S_W shifts only the base S1 power-law term, not the drop-kernel span S_k(t).",
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
