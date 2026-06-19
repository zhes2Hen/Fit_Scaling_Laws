# Three-stage MPL fitting

This folder contains the staged Multi-Power Law (MPL) fitting method used in
our final project.

The method keeps the original MPL functional form unchanged. Its contribution
is an optimization protocol that reduces compensation between the base power
law and the learning-rate decay kernel.

## Experimental protocol

- **Parameter fitting:** all reported model parameters are fitted using the
  cosine loss curve.
- **Protocol validation:** WSD is used during method development to select the
  late-decay fitting window.
- **Additional transfer test:** 8-1-1 is evaluated after the protocol is fixed
  and is not part of the final window-selection score.

This distinction is important: the final parameters use cosine observations
only, while WSD provides validation information for choosing the fitting
protocol.

## Why staged fitting?

Directly fitting all MPL parameters on cosine gives a good in-distribution fit,
but the base term and decay kernel can compensate for each other. The resulting
parameters transfer poorly to schedules with a stable phase or discrete
learning-rate drops.

We separate the signals that identify different parameter groups:

1. **Base isolation:** fit
   `L_base = L0 + A * S1^(-alpha)` on the first 30% of cosine, temporarily
   removing the decay kernel.
2. **Global calibration:** fix the Stage-1 exponent `alpha` and fit all other
   parameters on the complete post-warmup cosine trajectory.
3. **Decay calibration:** freeze `L0`, `A`, and `alpha`, then fit only the decay
   parameters on the final 6% of cosine.

The first 30% is a relatively smooth region that identifies the intrinsic
power-law trend. The final 6% concentrates the strongest relative
learning-rate-change signal while freezing the base prevents it from drifting.

## Requirements

Use Python 3.11 or newer. The parent repository already uses:

```bash
pip install numpy pandas scipy torch matplotlib
```

The dataset must be located at:

```text
loss curves/gpt_loss+lrs.pkl
```

## Run

From this directory:

```bash
python run_three_stage_mpl.py
```

For a quick smoke test:

```bash
python run_three_stage_mpl.py \
  --stage1-adam-steps 50 \
  --stage1-lbfgs-steps 5 \
  --adam-steps 50 \
  --lbfgs-steps 5 \
  --prefix smoke_test
```

The default run uses the settings from the reported experiment:

```text
fit start step     = 2048
fit stride         = 50
Stage 1 window     = first 30% of cosine
Stage 2 window     = full post-warmup cosine
Stage 3 window     = final 6% of cosine
Huber delta        = 1e-3 on log residuals
optimizer          = Adam followed by LBFGS
```

Outputs are written under:

```text
three-stage MPL fitting/outputs/
```

Each fitted stage saves parameters, optimizer information, metrics,
per-schedule predictions, and a three-panel curve figure.

## Reported results

Post-warmup RMSE on raw observations:

| Method | Cosine | WSD validation | 8-1-1 transfer |
|---|---:|---:|---:|
| Direct MPL | 0.0407 | 0.0604 | 0.0574 |
| Three-stage MPL | 0.0409 | 0.0409 | 0.0407 |

The detailed result and validation-window tables are stored in `results/`.

## Files

- `run_three_stage_mpl.py`: end-to-end reproducible pipeline.
- `staged_mpl.py`: base-only and parameter-freezing optimization components.
- `configs/`: initialization parameters used by the reported run.
- `results/`: report-ready metrics and final parameters.
- `figures/`: baseline and final curve comparisons.

## Limitations

- The 94% Stage-3 boundary is validation-sensitive and should be tested on more
  models and schedules.
- `stage2_initial_params.json` provides a stable numerical initialization from
  a preliminary cosine fit; Stage 1 overwrites its base parameters before the
  reported Stage-2 optimization.
- The decay-kernel parameters are not uniquely identifiable even when their
  predicted responses are similar.
- A stronger future protocol would replace manual boundaries with
  regularization, change-point detection, or bilevel validation.
