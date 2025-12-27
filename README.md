# Chronos Tunable

This package wraps Chronos 2 into a simple, tunable `nn.Module` with explicit
target, known-covariate, and unknown-covariate channels.

## Install

```bash
pip install -e .
```

Dependencies:
- `torch`
- `huggingface_hub`
- `chronos==2.2.2`

## Usage

```python
import torch
from ChronosTunable import get_chronos_model, pinball_loss

lookback = 512
lookahead = 96
target_dim = 1
known_cov_dim = 4
unknown_cov_dim = 2
ckpt_path = "~/chronos_ckpt"

model = get_chronos_model(
    lookback=lookback,
    lookahead=lookahead,
    target_dim=target_dim,
    known_covariate_dim=known_cov_dim,
    unknown_covariate_dim=unknown_cov_dim,
    ckpt_path=ckpt_path,
)

batch = 2
target_past = torch.randn(batch, lookback, target_dim)
known_past = torch.randn(batch, lookback, known_cov_dim)
known_future = torch.randn(batch, lookahead, known_cov_dim)
unknown_past = torch.randn(batch, lookback, unknown_cov_dim)

target_hat, unknown_hat = model(
    target_past=target_past,
    known_covariates_past=known_past,
    known_covariates_future=known_future,
    unknown_covariates_past=unknown_past,
)

# target_hat: [batch, lookahead, target_dim, num_quantiles]
target_future = torch.randn(batch, lookahead, target_dim)
mask = torch.isfinite(target_future)
loss = pinball_loss(target_hat, target_future, model.quantiles, mask=mask)
# This is the base training loss for target forecasts.
loss.backward()
```

## I/O Structure

Inputs:
- `target_past`: `[batch, lookback, target_dim]`  
  The historical values you want to forecast (e.g., load). Each channel is one target variable.
- `known_covariates_past`: `[batch, lookback, known_covariate_dim]`  
  Past values of covariates that are also available in the future (calendar features, weather forecasts).
- `known_covariates_future`: `[batch, lookahead, known_covariate_dim]`  
  Future values of those known covariates aligned with the forecast horizon.
- `unknown_covariates_past`: `[batch, lookback, unknown_covariate_dim]`  
  Past values of covariates that are *not* known in the future (latent drivers or exogenous signals).

Outputs:
- `target_hat`: `[batch, lookahead, target_dim, num_quantiles]`  
  Quantile forecasts for each target channel over the horizon.
- `unknown_hat`: `[batch, lookahead, unknown_covariate_dim, num_quantiles]` or `None`  
  Quantile forecasts for unknown covariate channels if you choose to model them.

Pinball loss:
- The pinball (quantile) loss is the standard loss for quantile forecasts.
- Use it for targets and unknown covariates by comparing their quantile predictions to the ground truth.
- Example for unknown covariates:
  ```python
  # If you also model unknown covariates, provide their future targets.
  # Here we create a fake target for demonstration only.
  unknown_future = torch.randn(batch, lookahead, unknown_cov_dim)

  # Start with the target loss computed above.
  total_loss = loss
  if unknown_hat is not None:
      unknown_loss = pinball_loss(unknown_hat, unknown_future, model.quantiles)
      total_loss = total_loss + unknown_loss
  ```

Checkpoint loading:
- `ckpt_path` points to a local directory. If it contains a valid Chronos 2 snapshot, it is used directly.
- If not present, a snapshot is downloaded from Hugging Face (`amazon/chronos-2`) into `ckpt_path`.
