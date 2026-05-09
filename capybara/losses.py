from __future__ import annotations

"""Losses for profile/count models.

This module exposes two API levels:

* ``MNLLLoss`` and ``log1pMSELoss`` are low-level losses. They expect tensors
  that have already been normalized or aggregated into the form the loss uses.
* ``profile_mnll_loss`` and ``count_log1p_mse_loss`` are training wrappers for
  CAPY/ProCapNet-style model outputs. They perform the profile/count-specific
  preprocessing before calling the low-level losses.
"""

import torch
from torch import Tensor


def MNLLLoss(pred_log_probs: Tensor, target_counts: Tensor) -> Tensor:
    """Low-level mean multinomial negative log-likelihood.

    Use this when predictions have already been converted to log
    probabilities, for example with ``torch.nn.functional.log_softmax``. For
    raw profile logits, use ``profile_mnll_loss`` instead.

    Parameters
    ----------
    pred_log_probs
        Predicted log probabilities with shape ``(batch, ...)``. All non-batch
        dimensions are flattened into one multinomial event per example.
    target_counts
        Observed counts with the same shape as ``pred_log_probs``.

    Returns
    -------
    Tensor
        Scalar mean negative log-likelihood.
    """
    pred_log_probs = pred_log_probs.reshape(pred_log_probs.shape[0], -1)
    target_counts = target_counts.reshape(target_counts.shape[0], -1)

    log_fact_sum = torch.lgamma(torch.sum(target_counts, dim=-1) + 1)
    log_prod_fact = torch.sum(torch.lgamma(target_counts + 1), dim=-1)
    log_prod_exp = torch.sum(target_counts * pred_log_probs, dim=-1)
    return -torch.mean(log_fact_sum - log_prod_fact + log_prod_exp)


def log1pMSELoss(pred_log_counts: Tensor, target_counts: Tensor) -> Tensor:
    """Low-level mean squared error on log-transformed counts.

    Use this when the target is already a total-count tensor. For profile
    labels that still need to be summed across strands and positions, use
    ``count_log1p_mse_loss`` instead.

    Parameters
    ----------
    pred_log_counts
        Predicted total counts in log space with shape ``(batch, ...)``.
    target_counts
        Observed total counts in count space with the same shape as
        ``pred_log_counts``.

    Returns
    -------
    Tensor
        Scalar mean squared error between ``pred_log_counts`` and
        ``log1p(target_counts)``.
    """
    return torch.nn.functional.mse_loss(pred_log_counts, torch.log1p(target_counts))


def profile_mnll_loss(
    pred_profile_logits: Tensor, target_profiles: Tensor, loss_mask: Tensor | None = None
) -> Tensor:
    """Training wrapper for profile multinomial loss.

    This accepts raw profile logits from a model, optionally applies a boolean
    loss mask, converts the included logits to log probabilities, and then
    calls ``MNLLLoss``. Use this in normal CAPY/ProCapNet training.

    Parameters
    ----------
    pred_profile_logits
        Predicted profile logits with shape ``(batch, strands, length)``.
    target_profiles
        Observed profile counts with shape ``(batch, strands, length)``.
    loss_mask
        Optional boolean mask with shape ``(batch, strands, length)``. ``True``
        positions are included in the profile loss.

    Returns
    -------
    Tensor
        Scalar mean profile negative log-likelihood.
    """
    if loss_mask is None:
        pred_log_probs = torch.nn.functional.log_softmax(
            pred_profile_logits.reshape(pred_profile_logits.shape[0], -1),
            dim=-1,
        )
        return MNLLLoss(pred_log_probs, target_profiles)

    losses = []
    for logits_i, target_i, mask_i in zip(pred_profile_logits, target_profiles, loss_mask):
        logits_masked = torch.masked_select(logits_i, mask_i)[None, :]
        target_masked = torch.masked_select(target_i, mask_i)[None, :]
        if target_masked.numel() == 0:
            continue
        pred_log_probs = torch.nn.functional.log_softmax(logits_masked, dim=-1)
        losses.append(MNLLLoss(pred_log_probs, target_masked))

    if not losses:
        return pred_profile_logits.sum() * 0
    return torch.stack(losses).mean()


def count_log1p_mse_loss(pred_log_counts: Tensor, target_profiles: Tensor) -> Tensor:
    """Training wrapper for total-count log1p MSE loss.

    This accepts predicted log total counts from a model and observed profile
    counts as labels. It sums observed profiles across strands and positions
    to produce total-count targets, then calls ``log1pMSELoss``. Use this in
    normal CAPY/ProCapNet training.

    Parameters
    ----------
    pred_log_counts
        Predicted total counts in log space with shape ``(batch, 1)``.
    target_profiles
        Observed profile counts with shape ``(batch, strands, length)``. Counts
        are summed over strands and positions before computing the loss.

    Returns
    -------
    Tensor
        Scalar mean squared error between predicted log counts and observed
        ``log1p`` total counts.
    """
    target_counts = target_profiles.sum(dim=(1, 2), keepdim=False).reshape(-1, 1)
    return log1pMSELoss(pred_log_counts, target_counts)
