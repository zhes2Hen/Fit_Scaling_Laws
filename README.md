# Learning Rate Schedule Scaling Law Fitting

This folder contains code for fitting several learning-rate-schedule scaling laws on loss curves.

The current workflow is:

1. Fit on the `cosine` learning rate schedule.
2. Extrapolate to `wsd` and `811`.
3. Save fitted parameters, metrics, predictions, and plots.

The code does not assume a fixed total number of steps, peak learning rate, or minimum learning rate. These values are read from the data.

## Python Requirements

Recommended Python version: Python 3.11 or newer.

Required packages:

```bash
pip install numpy pandas scipy torch matplotlib
```

Optional but useful:

```bash
pip install pyarrow
```

`pyarrow` may be needed depending on how your pandas installation reads pickle/table data.

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

## Warmup Shift: `S_W`

Some scaling laws use an intrinsic-time offset for a hidden warmup phase:

```text
S_W = peak_lr * s_w_prime
```

Each fitting script supports one command-line option:

```bash
--sw-mode fit
--sw-mode fixed
```

Meaning:

- `--sw-mode fit`: fit `s_w_prime` as a non-negative parameter.
- `--sw-mode fixed`: fix `s_w_prime = 0`.

If no option is given, the default is:

```bash
--sw-mode fit
```

Outputs are saved separately:

```text
fit_outputs/<method>_with_sw/
fit_outputs/<method>_no_sw/
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

Fits the Momentum Law, also related to the learning-rate-annealing scaling law.

The fitted form is:

```text
L_hat(t) = L0 + A * (S1(t) + S_W)^(-alpha) - C * S2(t)
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

Run without `S_W`:

```bash
python fit_momentum_law.py --sw-mode fixed
```

Run with fitted `S_W`:

```bash
python fit_momentum_law.py
```

or:

```bash
python fit_momentum_law.py --sw-mode fit
```

Outputs:

```text
fit_outputs/momentum_law_no_sw/
fit_outputs/momentum_law_with_sw/
```

### `fit_multi_power_law.py`

Fits the Multi-Power Law (MPL).

The fitted form is:

```text
L_hat(t) = L0 + A * (S1(t) + S_W)^(-alpha) - B * LD(t)
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
- the base power-law term can use `S_W`;
- the drop-kernel span `S_k(t)` is not shifted by `S_W`;
- parameters are constrained with smooth transforms.

Optimizer:

- PyTorch Adam;
- then PyTorch LBFGS;
- Huber loss on log residuals.

Run without `S_W`:

```bash
python fit_multi_power_law.py --sw-mode fixed
```

Run with fitted `S_W`:

```bash
python fit_multi_power_law.py
```

or:

```bash
python fit_multi_power_law.py --sw-mode fit
```

Outputs:

```text
fit_outputs/multi_power_law_no_sw/
fit_outputs/multi_power_law_with_sw/
```

### `fit_functional_scaling_law.py`

Fits the older Functional Scaling Laws reproduction route based on an `expected_R` style formula.

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
T_eff(t) = T(t) + peak_lr * s_w_prime
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

Run without `S_W`:

```bash
python fit_functional_scaling_law.py --sw-mode fixed
```

Run with fitted `S_W`:

```bash
python fit_functional_scaling_law.py
```

or:

```bash
python fit_functional_scaling_law.py --sw-mode fit
```

Outputs:

```text
fit_outputs/functional_scaling_law_no_sw/
fit_outputs/functional_scaling_law_with_sw/
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

Run all six fits:

```bash
python fit_momentum_law.py --sw-mode fixed
python fit_momentum_law.py --sw-mode fit

python fit_multi_power_law.py --sw-mode fixed
python fit_multi_power_law.py --sw-mode fit

python fit_functional_scaling_law.py --sw-mode fixed
python fit_functional_scaling_law.py --sw-mode fit
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

