from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from huggingface_hub import snapshot_download
from huggingface_hub.utils import LocalEntryNotFoundError


def _resolve_checkpoint(ckpt_path: Path, repo_id: str) -> str:
    ckpt_path = ckpt_path.expanduser()
    if ckpt_path.exists() and ckpt_path.is_dir():
        has_config = (ckpt_path / "config.json").exists()
        has_weights = any(ckpt_path.glob("*.safetensors")) or any(ckpt_path.glob("pytorch_model*.bin"))
        if has_config and has_weights:
            return str(ckpt_path)

    ckpt_path.mkdir(parents=True, exist_ok=True)
    try:
        return snapshot_download(
            repo_id=repo_id,
            cache_dir=str(ckpt_path),
            local_files_only=True,
        )
    except LocalEntryNotFoundError:
        return snapshot_download(
            repo_id=repo_id,
            cache_dir=str(ckpt_path),
        )


class ChronosTunableWrapper(nn.Module):
    def __init__(
        self,
        lookback: int,
        lookahead: int,
        target_dim: int,
        known_covariate_dim: int,
        unknown_covariate_dim: int,
        ckpt_path: Path,
        repo_id: str = "amazon/chronos-2",
    ) -> None:
        super().__init__()
        if lookback <= 0 or lookahead <= 0:
            raise ValueError("lookback and lookahead must be positive")
        if target_dim <= 0:
            raise ValueError("target_dim must be positive")
        if known_covariate_dim < 0 or unknown_covariate_dim < 0:
            raise ValueError("covariate dims must be non-negative")

        self.lookback = lookback
        self.lookahead = lookahead
        self.target_dim = target_dim
        self.known_covariate_dim = known_covariate_dim
        self.unknown_covariate_dim = unknown_covariate_dim

        local_dir = _resolve_checkpoint(Path(ckpt_path), repo_id)
        from chronos import Chronos2Pipeline

        self.pipe = Chronos2Pipeline.from_pretrained(local_dir)
        self.model = self.pipe.model
        self.output_patch_size = self.model.chronos_config.output_patch_size
        self.quantiles = list(self.model.chronos_config.quantiles)

        self.group_size = target_dim + unknown_covariate_dim + known_covariate_dim

    def forward(
        self,
        target_past: torch.Tensor,
        known_covariates_past: torch.Tensor | None,
        known_covariates_future: torch.Tensor | None,
        unknown_covariates_past: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if target_past.ndim != 3:
            raise ValueError("target_past must be [batch, lookback, target_dim]")
        if target_past.shape[1] != self.lookback or target_past.shape[2] != self.target_dim:
            raise ValueError("target_past shape does not match lookback/target_dim")

        if self.known_covariate_dim > 0:
            if known_covariates_past is None or known_covariates_future is None:
                raise ValueError("known covariates are required when known_covariate_dim > 0")
            if known_covariates_past.shape[1] != self.lookback or known_covariates_past.shape[2] != self.known_covariate_dim:
                raise ValueError("known_covariates_past shape mismatch")
            if (
                known_covariates_future.shape[1] != self.lookahead
                or known_covariates_future.shape[2] != self.known_covariate_dim
            ):
                raise ValueError("known_covariates_future shape mismatch")
        else:
            known_covariates_past = None
            known_covariates_future = None

        if self.unknown_covariate_dim > 0:
            if unknown_covariates_past is None:
                raise ValueError("unknown_covariates_past is required when unknown_covariate_dim > 0")
            if (
                unknown_covariates_past.shape[1] != self.lookback
                or unknown_covariates_past.shape[2] != self.unknown_covariate_dim
            ):
                raise ValueError("unknown_covariates_past shape mismatch")
        else:
            unknown_covariates_past = None

        batch_size = target_past.shape[0]
        device = target_past.device

        target_rows = target_past.transpose(1, 2)
        context_parts = [target_rows]

        if unknown_covariates_past is not None:
            context_parts.append(unknown_covariates_past.transpose(1, 2))
        if known_covariates_past is not None:
            context_parts.append(known_covariates_past.transpose(1, 2))

        context = torch.cat(context_parts, dim=1)
        context = context.reshape(batch_size * self.group_size, self.lookback)

        nan_rows = self.target_dim + self.unknown_covariate_dim
        future_pad = torch.full(
            (batch_size, nan_rows, self.lookahead),
            float("nan"),
            device=device,
        )
        future_parts = [future_pad]
        if known_covariates_future is not None:
            future_parts.append(known_covariates_future.transpose(1, 2))
        future_covariates = torch.cat(future_parts, dim=1)
        future_covariates = future_covariates.reshape(batch_size * self.group_size, self.lookahead)

        group_ids = torch.repeat_interleave(
            torch.arange(batch_size, device=device),
            repeats=self.group_size,
        )

        num_output_patches = math.ceil(self.lookahead / self.output_patch_size)
        outputs = self.model(
            context=context,
            future_covariates=future_covariates,
            group_ids=group_ids,
            num_output_patches=num_output_patches,
        )
        quantile_preds = outputs.quantile_preds[..., : self.lookahead]
        preds = quantile_preds.reshape(batch_size, self.group_size, -1, self.lookahead)
        preds = preds.permute(0, 1, 3, 2)

        target_hat = preds[:, : self.target_dim].permute(0, 2, 1, 3)
        unknown_hat = None
        if self.unknown_covariate_dim > 0:
            start = self.target_dim
            end = self.target_dim + self.unknown_covariate_dim
            unknown_hat = preds[:, start:end].permute(0, 2, 1, 3)

        return target_hat, unknown_hat


def get_chronos_model(
    lookback: int,
    lookahead: int,
    target_dim: int,
    known_covariate_dim: int,
    unknown_covariate_dim: int,
    ckpt_path: str | Path,
) -> ChronosTunableWrapper:
    return ChronosTunableWrapper(
        lookback=lookback,
        lookahead=lookahead,
        target_dim=target_dim,
        known_covariate_dim=known_covariate_dim,
        unknown_covariate_dim=unknown_covariate_dim,
        ckpt_path=Path(ckpt_path),
    )


def pinball_loss(
    preds: torch.Tensor,
    target: torch.Tensor,
    quantiles: Iterable[float],
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if preds.ndim != 4:
        raise ValueError("preds must be [batch, horizon, channels, num_quantiles]")
    if target.ndim != 3:
        raise ValueError("target must be [batch, horizon, channels]")
    if preds.shape[:3] != target.shape:
        raise ValueError("preds and target shapes are incompatible")

    q = torch.tensor(list(quantiles), device=preds.device, dtype=preds.dtype)
    diff = target.unsqueeze(-1) - preds
    loss = torch.maximum(q * diff, (q - 1.0) * diff)

    if mask is not None:
        if mask.shape != target.shape:
            raise ValueError("mask must match target shape")
        loss = loss[mask.unsqueeze(-1)]
        return loss.mean() if loss.numel() > 0 else loss.new_tensor(0.0)

    return loss.mean()
