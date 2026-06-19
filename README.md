# Learning Rate Schedule Scaling Law Fitting

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository fits and compares learning-rate-schedule scaling laws for LLM pre-training loss curves. The scripts fit on the `cosine` schedule, then evaluate transfer to `wsd` and `811`.

## Python Requirements

Recommended Python version: Python 3.11 or newer.

Required packages:

```bash
pip install numpy pandas scipy torch matplotlib pyarrow
```

CUDA is optional. PyTorch scripts use CUDA when it is available and fall back to CPU otherwise.

## Data

The expected data file is:

```text
loss curves/gpt_loss+lrs.pkl
```

Each schedule data frame should contain:

```text
step
lr
Metrics/loss
```

The utility file expects these schedule names:

```python
SCHEDULES = ["cosine", "wsd", "811"]
```

If your data uses different names, update `SCHEDULES` and `schedule_key()` in `scaling_fit_utils.py`.

## Shared Fitting Details

All scripts use:

- raw loss for fitting, not smoothed loss;
- Huber loss on log-loss residuals;
- Huber delta `1e-3`;
- `cosine` as the main training schedule;
- `wsd` and `811` as transfer schedules;
- smoothed curves only for figures;
- output under `fit_outputs/`.

The residual is:

```text
log(observed_loss) - log(predicted_loss)
```

The main metric window is post-warmup:

```text
step >= 2048
```

The LR accumulation still starts from step `0`. Only the fitted/evaluated residual window starts at step `2048`.

## Missing-Step Handling

The loader allows at most one missing step per schedule.

If one step is missing:

- loss is not interpolated;
- the missing loss is stored as `NaN`;
- metrics skip the missing loss point;
- LR is filled by linear interpolation from the observed LR column;
- cumulative LR quantities use the full reconstructed LR sequence.

If more than one step is missing, the code raises an error.

## Files

| File | Purpose |
|---|---|
| `scaling_fit_utils.py` | Data loading, missing-step checks, metrics, smoothing, and output writing. |
| `fit_momentum_law.py` | Momentum Law / MTL baseline. |
| `fit_multi_power_law.py` | Direct Multi-Power Law (MPL) fit. |
| `fit_functional_scaling_law.py` | FSL expected-R style fit. |
| `fit_elr_fsl.py` | ELR-FSL with fixed-step optimization. |
| `fit_staged_elr_functional_scaling_law.py` | Staged ELR-FSL protocol. |
| `fit_staged_multi_power_law.py` | Staged MPL protocol. |
| `staged_multi_power_law.py` | Helper module used by staged MPL. |
| `fit_staged_elr_multi_power_law.py` | Staged ELR-MPL protocol. |

## 1. Momentum Law

Run:

```bash
python fit_momentum_law.py
```

Optional warmup-shift experiment:

```bash
python fit_momentum_law.py --sw-mode fit
```

Default output:

```text
fit_outputs/momentum_law_no_sw/
```

Formula:

```text
L_hat(t) = L0 + A * S1(t)^(-alpha) - C * S2(t)
S1(t) = sum_{i=0}^t eta_i
m_i = lambda * m_{i-1} + (eta_{i-1} - eta_i)
S2(t) = sum_{i=1}^t m_i
```

Default `lambda = 0.999`. This script uses SciPy `L-BFGS-B` with several initial points.

## 2. Direct Multi-Power Law

Run:

```bash
python fit_multi_power_law.py
```

Optional warmup-shift experiment:

```bash
python fit_multi_power_law.py --sw-mode fit
```

Default output:

```text
fit_outputs/multi_power_law_no_sw/
```

Formula:

```text
L_hat(t) = L0 + A * S1(t)^(-alpha) - B * LD(t)
LD(t) = sum_{k <= t} drop_k * G(eta_k^(-gamma) * S_k(t))
drop_k = max(eta_{k-1} - eta_k, 0)
S_k(t) = sum_{j=k}^t eta_j
G(x) = 1 - (1 + C * x)^(-beta)
```

The code precomputes an `MPLCache` tensor for active LR drops. It uses Adam followed by LBFGS.

## 3. Functional Scaling Law

Run:

```bash
python fit_functional_scaling_law.py
```

Optional warmup-shift experiment:

```bash
python fit_functional_scaling_law.py --sw-mode fit
```

Default output:

```text
fit_outputs/functional_scaling_law_no_sw/
```

This script follows the expected-R style FSL route used in the old reproduction code. It is different from the subtractive fitting method in the FSL paper appendix.

Formula:

```text
L_hat(k)
  = c1 * M^(-s * beta)
  + c2 * tau_k^(-s)
  + c3 * N_mini(k)
  + c4 * N_label(k)

tau_k = integral_0^k eta(r) dr
N_mini(k)  = integral K(tau_k - tau(r)) * E_beta,s(tau(r)) * eta_floor(r)^2 dr
N_label(k) = integral K(tau_k - tau(r)) * sigma^2 * eta_floor(r)^2 dr
K(u) = (1 + u)^(-2 + 1 / beta)
```

The finite-width energy term is:

```text
E_beta,s(u) = integral from M^(-beta) to 1 of z^(s-1) * exp(-2 z u) dz
```

The script uses Adam followed by LBFGS.

## 4. ELR-FSL

Run the default lambda-zero setting:

```bash
python fit_elr_fsl.py
```

Equivalent default:

```bash
python fit_elr_fsl.py --lambda-mode fixed --lambda-value 0
```

Fit lambda:

```bash
python fit_elr_fsl.py --lambda-mode fit
```

Fix lambda to another value:

```bash
python fit_elr_fsl.py --lambda-mode fixed --lambda-value 0.01
```

Default output:

```text
fit_outputs/elr_functional_scaling_law_lambda_zero/
```

ELR-FSL first fits a norm-square recurrence:

```text
n_{k+1} = (1 - eta_k * lambda)^2 * n_k + eta_k^2 * C_u
ELR_k = eta_k / sqrt(n_k)
tau_k = sum_{i < k} ELR_i
```

Then it uses the same FSL form as above, replacing the raw LR by `ELR`.

This script uses fixed optimizer lengths:

```text
fit stride = 50
Adam steps = 2200
LBFGS max_iter = 70
```

## 5. Staged ELR-FSL

Run:

```bash
python fit_staged_elr_functional_scaling_law.py
```

Useful optional arguments:

```bash
python fit_staged_elr_functional_scaling_law.py \
  --fit-stride 20 \
  --oracle-decay-weights 10,20 \
  --stage1-end-fractions 0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.60 \
  --stage3-starts 0.925,0.935,0.94,0.945,0.955
```

Main output summary:

```text
fit_outputs/staged_elr_functional_scaling_law_summary.csv
```

This method uses the lambda-zero ELR-FSL formula, but fits it in stages:

1. Fit a weighted cosine+WSD model to estimate a useful signal exponent.
2. Scan short cosine prefix windows and choose the prefix whose exponent best matches that estimate.
3. Refit the full cosine curve with selected parameters fixed.
4. Fit tail candidates and choose the one with the best WSD post-2048 RMSE.

The final parameters are still fitted through the staged protocol; WSD is used for protocol validation and candidate selection.

## 6. Staged Multi-Power Law

Run:

```bash
python fit_staged_multi_power_law.py
```

Smoke test:

```bash
python fit_staged_multi_power_law.py \
  --stage1-adam-steps 50 \
  --stage1-lbfgs-steps 5 \
  --adam-steps 50 \
  --lbfgs-steps 5 \
  --prefix smoke_staged_multi_power_law
```

Default output folders:

```text
fit_outputs/staged_multi_power_law_stage1_base/
fit_outputs/staged_multi_power_law_stage2_full_cosine/
fit_outputs/staged_multi_power_law_stage3_decay/
```

The MPL formula is unchanged. The difference is the fitting protocol:

1. Fit the base term on the first 30% of cosine:

```text
L_base(t) = L0 + A * S1(t)^(-alpha)
```

2. Fix `alpha` and fit the full MPL on all post-warmup cosine points.
3. Fix `L0`, `A`, and `alpha`, then fit the decay-kernel parameters on the last 6% of cosine.

Default settings:

```text
fit start step = 2048
fit stride = 50
stage1 fraction = 0.30
stage3 fraction = 0.94
optimizer = Adam followed by LBFGS
```

`staged_multi_power_law.py` is the helper module for this script.

## 7. Staged ELR-MPL

Run:

```bash
python fit_staged_elr_multi_power_law.py
```

Useful optional arguments:

```bash
python fit_staged_elr_multi_power_law.py \
  --fit-stride 20 \
  --oracle-decay-weights 1,10,20 \
  --stage1-end-fractions 0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.60 \
  --stage3-start-fractions 0.90,0.925,0.935,0.94,0.945,0.95,0.955,0.96 \
  --norm-initialization-grid 1e-4:1,1:10,100:1,100:10,10000:10
```

Main output summary:

```text
fit_outputs/staged_elr_multi_power_law_summary.csv
```

This method keeps the MPL drop-kernel structure but replaces raw LR by ELR:

```text
n_{k+1} = n_k + eta_k^2 * C_u
ELR_k = eta_k / sqrt(n_k)
S1_ELR(t) = sum_{i=0}^t ELR_i
```

Then:

```text
L_hat(t) = L0 + A * S1_ELR(t)^(-alpha) - B * LD_ELR(t)
LD_ELR(t) = sum_{k <= t} drop_ELR_k * G(ELR_k^(-gamma) * S_k_ELR(t))
```

The staged fitting protocol follows the staged MPL idea, with extra scans for the ELR norm parameters `C_u` and `n0`. It uses Adam/LBFGS early stopping rather than fixed iteration budgets.

## Output Files

Each method folder usually contains:

```text
params.json
fit_info.json
metrics.csv
cosine_predictions.csv
wsd_predictions.csv
811_predictions.csv
fit_curves.png
```

Some ELR scripts also save:

```text
cosine_elr_dynamics.csv
wsd_elr_dynamics.csv
811_elr_dynamics.csv
```

The shared data diagnostics file is:

```text
fit_outputs/data_diagnostics.json
```

## Recommended Run Order

Basic reproduction:

```bash
python fit_momentum_law.py
python fit_multi_power_law.py
python fit_functional_scaling_law.py
python fit_elr_fsl.py
```

Developed methods:

```bash
python fit_staged_elr_functional_scaling_law.py
python fit_staged_multi_power_law.py
python fit_staged_elr_multi_power_law.py
```

Then compare:

```text
fit_outputs/*/metrics.csv
fit_outputs/*/params.json
fit_outputs/*/fit_curves.png
```

## Notes

- The code infers step count, peak LR, and missing LR values from the data.
- Fitting is done on raw loss. Plot smoothing is only for visualization.
- `--sw-mode` defaults to `fixed`, so `S_W = 0` unless explicitly enabled.
- The staged protocols use WSD during method development or candidate selection. Treat their transfer metrics with that validation role in mind.
