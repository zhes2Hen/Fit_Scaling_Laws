# Learning Rate Schedule Scaling Law Fitting

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository reproduces and compares three published learning-rate-schedule (LRS) scaling law methods for LLM pre-training loss curves, and proposes our own method, as part of the course project *Topics in Deep Learning Theory (Spring 2026), Task 2: Predicting Loss Curves of LLM Pre-training*.

## Methods

| Script | Method | Reference |
|--------|--------|-----------|
| `fit_momentum_law.py` | Momentum Law (MTL) | [Tissue et al., 2024 — arXiv:2408.11029](https://arxiv.org/abs/2408.11029) |
| `fit_multi_power_law.py` | Multi-Power Law (MPL) | [Luo et al., 2024 — arXiv:2503.12811](https://arxiv.org/abs/2503.12811) |
| `fit_functional_scaling_law.py` | Functional Scaling Law (FSL, old-notebook route) | [Li et al., 2025 — arXiv:2509.19189](https://arxiv.org/abs/2509.19189) |
| `fit_our_method.py` *(XXX)* | Our Method *(XXX)* | — |


## Workflow

1. Fit each method on the `cosine` LRS only.
2. Extrapolate to `wsd` and `811` without re-fitting.
3. Evaluate on all observed steps; the primary metric window is `step ≥ 2048` (post-warmup).
4. Save fitted parameters, metrics, per-step predictions, and plots.

The code infers the full step range, peak LR, and minimum LR directly from each curve's data columns — nothing is hard-coded.

## Python Requirements

Recommended Python version: Python 3.11 or newer.

Required packages:

```bash
pip install numpy pandas scipy torch matplotlib pyarrow
```

GPU is optional. The PyTorch-based fits use CUDA if it is available, otherwise they fall back to CPU.

## Data Format

The data file is expected at:

```text
loss curves/gpt_loss+lrs.pkl
```

The file should contain pandas data frames with keys like:

```text
M:100M_gpt_D:20B_scheduler:cosine_rope
M:100M_gpt_D:20B_scheduler:wsd_rope
M:100M_gpt_D:20B_scheduler:811_rope
```

Each data frame should contain at least these columns:

```text
step
lr
Metrics/loss
```

The shared utility code currently uses these schedule names:

```python
SCHEDULES = ["cosine", "wsd", "811"]
```

If you use different schedule names, update `SCHEDULES` and `schedule_key()` in `scaling_fit_utils.py`.

## Missing-Step Handling

The code allows at most one missing step per schedule.

If one step is missing:

- the loss is not interpolated;
- the missing loss is stored as `NaN`;
- metrics skip that missing loss point;
- the learning rate is filled by linear interpolation from the observed `lr` column;
- all cumulative LR quantities still use the full LR sequence.

If more than one step is missing, the code raises an error.

## Common Fitting Details

All three methods use the same basic fitting setup:

- train only on `cosine`;
- extrapolate to `wsd` and `811`;
- fit on raw loss values, not smoothed loss;
- use Huber loss on log loss residuals;
- Huber delta is `1e-3`;
- save both all-observed metrics and post-2048 metrics;
- use smoothed curves only for plotting.

The log residual is:

```text
log(observed_loss) - log(predicted_loss)
```

The main output folder is:

```text
fit_outputs/
```

## File Overview

### `scaling_fit_utils.py`

Shared utility functions.

It handles:

- loading the loss curve pickle;
- checking missing steps;
- filling missing LR values;
- computing metrics;
- smoothing curves for plots;
- saving parameters, predictions, metrics, and figures.

This file does not fit a model by itself.

### `fit_momentum_law.py`

Fits the Momentum Law from [arXiv:2408.11029](https://arxiv.org/pdf/2408.11029).

The fitted form is:

```text
L_hat(t) = L0 + A * S1(t)^(-alpha) - C * S2(t)
```

where:

```text
S1(t) = sum_{i=0}^t lr_i
```

The momentum term is:

```text
m_i = lambda * m_{i-1} + (lr_{i-1} - lr_i)
S2(t) = sum_{i=1}^t m_i
```

Current setting:

```text
lambda = 0.999
```

Optimizer:

- SciPy `L-BFGS-B`;
- grid search over several initial values;
- Huber loss on log residuals.

Run:

```bash
python fit_momentum_law.py
```

Output:

```text
fit_outputs/momentum_law_no_sw/
```

### `fit_multi_power_law.py`

Fits the Multi-Power Law (MPL) from [arXiv:2503.12811](https://arxiv.org/pdf/2503.12811).

The fitted form is:

```text
L_hat(t) = L0 + A * S1(t)^(-alpha) - B * LD(t)
```

The loss-drop term is:

```text
LD(t) = sum_{k <= t} drop_k * G(lr_k^(-gamma) * S_k(t))
```

where:

```text
drop_k = max(lr_{k-1} - lr_k, 0)
S_k(t) = sum_{j=k}^t lr_j
G(x) = 1 - (1 + C * x)^(-beta)
```

Technical notes:

- only positive LR drops are used;
- the implementation precomputes an `MPLCache` tensor for speed;
- parameters are constrained with smooth transforms.

Optimizer:

- PyTorch Adam;
- then PyTorch LBFGS;
- Huber loss on log residuals.

Run:

```bash
python fit_multi_power_law.py
```

Output:

```text
fit_outputs/multi_power_law_no_sw/
```

### `fit_functional_scaling_law.py`

Fits the Functional Scaling Laws (FSL) from [arXiv:2509.19189](https://arxiv.org/pdf/2509.19189).

This is not the final practical LLM ansatz from the FSL paper. It follows the older notebook-style decomposition:

```text
R_hat(t) =
    c1 * M^(-s * beta)
  + c2 * T_eff(t)^(-s)
  + c3 * minibatch_noise(t)
  + c4 * label_noise(t)
```

The effective intrinsic time is:

```text
T_eff(t) = T(t)
T(t) = integral_0^t lr(r) dr
```

The two noise terms are computed with numerical quadrature.

Technical notes:

- `FSLCache` stores LR and intrinsic-time information;
- Simpson-style quadrature is used for the signal integral;
- trapezoid-style quadrature is used for the noise integrals;
- PyTorch autograd is used for trainable parameters.

Optimizer:

- PyTorch Adam;
- then PyTorch LBFGS;
- Huber loss on log residuals.

Run:

```bash
python fit_functional_scaling_law.py
```

Output:

```text
fit_outputs/functional_scaling_law_no_sw/
```

## Output Files

Each method output folder contains:

```text
params.json
fit_info.json
metrics.csv
cosine_predictions.csv
wsd_predictions.csv
811_predictions.csv
fit_curves.png
```

Meaning:

- `params.json`: fitted parameters;
- `fit_info.json`: optimizer settings, runtime, fitting details;
- `metrics.csv`: metrics for all schedules;
- `*_predictions.csv`: per-step LR, observed loss, and predicted loss;
- `fit_curves.png`: three-panel plot for cosine, WSD, and 811.

The shared data diagnostics file is:

```text
fit_outputs/data_diagnostics.json
```

It records:

- row counts;
- inferred full step count;
- missing steps;
- peak LR from the data;
- LR values around missing steps.

## Recommended Run Order

```bash
python fit_momentum_law.py
python fit_multi_power_law.py
python fit_functional_scaling_law.py
```

Then compare:

```text
fit_outputs/*/metrics.csv
fit_outputs/*/params.json
fit_outputs/*/fit_curves.png
```

## Notes for Future Changes

- To fit a different training schedule, change the training schedule name inside each fitting script.
- To add more schedules, update `SCHEDULES` in `scaling_fit_utils.py`.
- To use a different data file, update `DATA_PATH` in `scaling_fit_utils.py`.
- Fitting is done on raw loss. Do not smooth the loss before fitting unless you intentionally want a different objective.
- Plot smoothing is only for visualization.
