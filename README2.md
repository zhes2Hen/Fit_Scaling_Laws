# Extra FSL and ELR-FSL Fitting Scripts

This file describes three extra scripts:

- `fit_elr_fsl.py`
- `fit_fsl_develop.py`
- `fit_elr_fsl_develop.py`

They use the same project data format and the same shared utilities as the main README.
They fit only on the `cosine` loss curve and extrapolate to `wsd` and `811`.

## Python Requirements

Recommended Python version: Python 3.11 or newer.

Required packages:

```bash
pip install numpy pandas scipy torch matplotlib pyarrow
```

CUDA is optional. If CUDA is available, the PyTorch scripts use it. Otherwise they run on CPU.

## Data

The expected data file is:

```text
loss curves/gpt_loss+lrs.pkl
```

The expected schedules are:

```python
SCHEDULES = ["cosine", "wsd", "811"]
```

The shared loader is in `scaling_fit_utils.py`.
It fills at most one missing learning-rate step, but it does not fill missing loss values.
Metrics skip missing loss values.

## Common Details

All three scripts use:

- raw loss for fitting;
- Huber loss on log-loss residuals;
- Huber delta `1e-3`;
- `cosine` as the only training schedule;
- `wsd` and `811` as extrapolation schedules;
- smoothed curves only for plotting;
- output under `fit_outputs/`.

The log residual is:

```text
log(observed_loss) - log(predicted_loss)
```

## `fit_elr_fsl.py`

This script fits ELR-FSL with a fixed number of optimizer steps.

Run the default setting:

```bash
python fit_elr_fsl.py
```

The default is:

```text
--lambda-mode fixed --lambda-value 0
```

So the default output folder is:

```text
fit_outputs/elr_functional_scaling_law_lambda_zero/
```

To fit lambda:

```bash
python fit_elr_fsl.py --lambda-mode fit
```

To fix lambda to another value:

```bash
python fit_elr_fsl.py --lambda-mode fixed --lambda-value 0.01
```

### Formula

The script first defines a fitted norm-square recurrence:

```text
n_{k+1} = (1 - eta_k * lambda)^2 * n_k + eta_k^2 * C_u
```

Then it defines the effective learning rate:

```text
ELR_k = eta_k / sqrt(n_k)
```

The intrinsic time is:

```text
tau_k = sum_{i < k} ELR_i
```

The prediction has the same FSL form:

```text
L_hat(k)
  = c1 * M^(-s * beta)
  + c2 * tau_k^(-s)
  + c3 * N_mini(k)
  + c4 * N_label(k)
```

where:

```text
N_mini(k)  = integral K(tau_k - tau(r)) * E_beta,s(tau(r)) * ELR_floor(r)^2 dr
N_label(k) = integral K(tau_k - tau(r)) * sigma^2 * ELR_floor(r)^2 dr
```

The kernel is:

```text
K(u) = (1 + u)^(-2 + 1 / beta)
```

The finite-width energy term is:

```text
E_beta,s(u) = integral from M^(-beta) to 1 of z^(s-1) * exp(-2 z u) dz
```

### Optimizer

This script uses fixed optimizer lengths:

- Adam for `2200` steps;
- PyTorch LBFGS with `max_iter=70`;
- fit stride `50`;
- fit starts at step `2048`.

It saves:

- `params.json`;
- `fit_info.json`;
- `metrics.csv`;
- per-schedule prediction CSV files;
- per-schedule ELR dynamics CSV files;
- `fit_curves.png`.

## `fit_fsl_develop.py`

This is a develop version of FSL.
It does not use `S_W`.
It keeps the FSL formula, but exposes stopping and stride settings as argparse options.

Run the default setting:

```bash
python fit_fsl_develop.py
```

Default output:

```text
fit_outputs/functional_scaling_law_develop_stride20_tight/
```

### Formula

The intrinsic time is based on the original learning rate:

```text
tau_k = sum_{i < k} eta_i
```

The prediction is:

```text
L_hat(k)
  = c1 * M^(-s * beta)
  + c2 * tau_k^(-s)
  + c3 * N_mini(k)
  + c4 * N_label(k)
```

with:

```text
N_mini(k)  = integral K(tau_k - tau(r)) * E_beta,s(tau(r)) * eta_floor(r)^2 dr
N_label(k) = integral K(tau_k - tau(r)) * sigma^2 * eta_floor(r)^2 dr
```

This is the old-notebook FSL route. It is not the subtractive fitting method from the FSL paper appendix.

### Main Arguments

```bash
python fit_fsl_develop.py \
  --fit-stride 20 \
  --adam-max-steps 2000 \
  --adam-min-steps 400 \
  --adam-patience 250 \
  --adam-min-delta 5e-9 \
  --lbfgs-max-steps 4 \
  --lbfgs-inner-iter 30 \
  --lbfgs-patience 3 \
  --lbfgs-min-delta 1e-9 \
  --grad-tol 1e-7 \
  --output-suffix stride20_tight
```

The defaults are the same as the command above.

### Stopping Rule

Adam stops after `adam-min-steps` if either:

- the best loss has not improved by `adam-min-delta` for `adam-patience` steps;
- the max absolute gradient is below `grad-tol`.

LBFGS is called several times.
After each outer call, the script checks whether the loss improved by `lbfgs-min-delta`.
It stops if there is no improvement for `lbfgs-patience` outer calls, or if the max gradient is below `grad-tol`.

## `fit_elr_fsl_develop.py`

This is the develop version of ELR-FSL.
It combines:

- the ELR-FSL norm recurrence from `fit_elr_fsl.py`;
- the stopping and stride arguments from `fit_fsl_develop.py`.

Run the default setting:

```bash
python fit_elr_fsl_develop.py
```

The default is:

```text
--lambda-mode fixed --lambda-value 0
```

Default output:

```text
fit_outputs/elr_functional_scaling_law_lambda_zero_develop_stride20_tight/
```

To fit lambda:

```bash
python fit_elr_fsl_develop.py --lambda-mode fit
```

To fix lambda to another value:

```bash
python fit_elr_fsl_develop.py --lambda-mode fixed --lambda-value 0.01
```

All stopping arguments are the same as in `fit_fsl_develop.py`.

### Formula

The norm-square recurrence is:

```text
n_{k+1} = (1 - eta_k * lambda)^2 * n_k + eta_k^2 * C_u
```

The effective learning rate is:

```text
ELR_k = eta_k / sqrt(n_k)
```

The script then uses the same FSL prediction as above, but replaces every learning-rate schedule inside intrinsic time and noise accumulation by `ELR`.

When `--lambda-mode fixed --lambda-value 0`, the recurrence becomes:

```text
n_{k+1} = n_k + eta_k^2 * C_u
```

This setting is the default because previous experiments found that fitted or large positive lambda can overfit the cosine curve and hurt WSD/811 extrapolation.

## Output Naming

For `fit_elr_fsl.py`:

```text
lambda fixed 0      -> elr_functional_scaling_law_lambda_zero
lambda fit          -> elr_functional_scaling_law_free_lambda
lambda fixed 0.01   -> elr_functional_scaling_law_lambda_fixed_0p01
```

For `fit_elr_fsl_develop.py`:

```text
lambda fixed 0      -> elr_functional_scaling_law_lambda_zero_develop_stride20_tight
lambda fit          -> elr_functional_scaling_law_free_lambda_develop_stride20_tight
lambda fixed 0.01   -> elr_functional_scaling_law_lambda_fixed_develop_0p01_stride20_tight
```

The last part comes from `--output-suffix`.

## Notes

- These scripts do not contain `from __future__ import annotations`.
- `fit_fsl_develop.py` and `fit_elr_fsl_develop.py` are meant for development experiments.
- `fit_elr_fsl.py` is closer to the original fixed-step ELR-FSL experiment.
- The shared `--sw-mode` option lives in `scaling_fit_utils.py`; these three scripts do not use `S_W`.
