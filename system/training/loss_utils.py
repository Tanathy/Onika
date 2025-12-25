# pyright: reportPrivateImportUsage=false
# pyright: reportAttributeAccessIssue=false
"""
Advanced loss utilities for LoRA training.
Inspired by Kohya sd-scripts custom_train_functions.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Union
from diffusers import DDPMScheduler


def conditional_loss(
    model_pred: torch.Tensor,
    target: torch.Tensor,
    reduction: str = "mean",
    loss_type: str = "l2",
    huber_c: float = 0.1,
) -> torch.Tensor:
    """
    Compute loss based on loss_type.
    
    Args:
        model_pred: Model prediction
        target: Target tensor
        reduction: "mean", "sum", or "none"
        loss_type: "l2", "huber", or "smooth_l1"
        huber_c: Huber loss C parameter (delta)
    
    Returns:
        Loss tensor
    """
    if loss_type == "l2":
        loss = F.mse_loss(model_pred, target, reduction=reduction)
    elif loss_type == "huber":
        loss = F.huber_loss(model_pred, target, reduction=reduction, delta=huber_c)
    elif loss_type == "smooth_l1":
        loss = F.smooth_l1_loss(model_pred, target, reduction=reduction)
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")
    
    return loss


def compute_snr(
    noise_scheduler: DDPMScheduler,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    """
    Compute SNR (Signal-to-Noise Ratio) for given timesteps.
    
    From the diffusers implementation.
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(device=timesteps.device, dtype=torch.float32)
    sqrt_alphas_cumprod = alphas_cumprod ** 0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5
    
    sqrt_alphas_cumprod = sqrt_alphas_cumprod[timesteps].float()
    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[timesteps].float()
    
    snr = (sqrt_alphas_cumprod / sqrt_one_minus_alphas_cumprod) ** 2
    return snr


def apply_snr_weight(
    loss: torch.Tensor,
    timesteps: torch.Tensor,
    noise_scheduler: DDPMScheduler,
    gamma: float = 5.0,
    v_parameterization: bool = False,
) -> torch.Tensor:
    """
    Apply Min-SNR weighting to the loss.
    
    From https://arxiv.org/abs/2303.09556
    Reduces high-timestep loss contribution for more stable training.
    
    Args:
        loss: Per-sample loss tensor [batch_size]
        timesteps: Timesteps tensor
        noise_scheduler: DDPM scheduler
        gamma: SNR gamma (typically 5.0)
        v_parameterization: Whether using v-prediction
    
    Returns:
        Weighted loss tensor
    """
    snr = compute_snr(noise_scheduler, timesteps)
    
    if v_parameterization:
        # For v-prediction: SNR / (SNR + 1)
        mse_loss_weights = snr / (snr + 1)
    else:
        # For epsilon prediction: min(SNR, gamma) / SNR
        mse_loss_weights = torch.stack([snr, gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
    
    loss = loss * mse_loss_weights
    return loss


def scale_v_prediction_loss_like_noise_prediction(
    loss: torch.Tensor,
    timesteps: torch.Tensor,
    noise_scheduler: DDPMScheduler,
) -> torch.Tensor:
    """
    Scale v-prediction loss to be similar to noise prediction loss.
    
    This helps stabilize training when using v-prediction.
    From Kohya sd-scripts.
    """
    snr = compute_snr(noise_scheduler, timesteps)
    scale = snr / (snr + 1)
    loss = loss * scale
    return loss


def add_v_prediction_like_loss(
    loss: torch.Tensor,
    timesteps: torch.Tensor,
    noise_scheduler: DDPMScheduler,
    v_pred_like_loss: float = 0.0,
) -> torch.Tensor:
    """
    Add v-prediction-like loss component.
    
    This adds a scaled component based on SNR to make epsilon prediction
    behave more like v-prediction at high noise levels.
    
    Args:
        loss: Per-sample loss tensor
        timesteps: Timesteps tensor
        noise_scheduler: DDPM scheduler
        v_pred_like_loss: Strength (0.0 = disabled)
    
    Returns:
        Modified loss tensor
    """
    if v_pred_like_loss <= 0.0:
        return loss
    
    snr = compute_snr(noise_scheduler, timesteps)
    v_lambda = v_pred_like_loss * snr / (snr + 1)
    loss = loss * (1 + v_lambda)
    return loss


def apply_debiased_estimation(
    loss: torch.Tensor,
    timesteps: torch.Tensor,
    noise_scheduler: DDPMScheduler,
) -> torch.Tensor:
    """
    Apply debiased estimation to the loss.
    
    From Kohya sd-scripts. Compensates for timestep distribution bias.
    """
    snr = compute_snr(noise_scheduler, timesteps)
    
    # Debiased estimation weight: 1 / sqrt(1 + 1/SNR)
    weight = 1.0 / (1.0 + 1.0 / snr).sqrt()
    loss = loss * weight
    return loss


def apply_masked_loss(
    loss: torch.Tensor,
    mask: Optional[torch.Tensor],
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Apply mask to loss (e.g., from alpha channel).
    
    Args:
        loss: Loss tensor [batch, channels, height, width] or [batch]
        mask: Optional mask tensor with same spatial dims
        reduction: How to reduce the masked loss
    
    Returns:
        Masked loss tensor
    """
    if mask is None:
        return loss
    
    # Ensure mask has same shape
    if len(loss.shape) == 4 and len(mask.shape) == 3:
        mask = mask.unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]
    
    if len(loss.shape) == 4:
        # Spatial loss
        loss = loss * mask
        if reduction == "mean":
            # Mean over spatial dimensions, weighted by mask
            loss = loss.sum(dim=[1, 2, 3]) / (mask.sum(dim=[1, 2, 3]) + 1e-8)
        elif reduction == "sum":
            loss = loss.sum(dim=[1, 2, 3])
    
    return loss


def fix_noise_scheduler_betas_for_zero_terminal_snr(noise_scheduler: DDPMScheduler):
    """
    Fix the noise scheduler betas to achieve zero terminal SNR.
    
    This modifies the scheduler in-place. From Kohya sd-scripts.
    Reference: https://arxiv.org/abs/2305.08891
    """
    alphas = 1.0 - noise_scheduler.betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    # Enforce zero terminal SNR
    alphas_bar_sqrt = alphas_cumprod.sqrt()
    
    # Shift so last value is 0
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()
    
    alphas_bar_sqrt -= alphas_bar_sqrt_T
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)
    
    # Recompute alphas_cumprod
    alphas_cumprod = alphas_bar_sqrt ** 2
    
    # Ensure values are valid
    alphas_cumprod[-1] = 1e-6  # Small but not zero to avoid numerical issues
    
    # Recompute alphas and betas
    alphas = torch.zeros_like(alphas_cumprod)
    alphas[0] = alphas_cumprod[0]
    alphas[1:] = alphas_cumprod[1:] / alphas_cumprod[:-1]
    
    betas = 1.0 - alphas
    
    # Update scheduler (in-place)
    noise_scheduler.betas = betas
    noise_scheduler.alphas = alphas
    noise_scheduler.alphas_cumprod = alphas_cumprod


def prepare_scheduler_for_custom_training(
    noise_scheduler: DDPMScheduler,
    device: torch.device,
    enable_zero_terminal_snr: bool = False,
):
    """
    Prepare noise scheduler for custom training with SNR-related features.
    
    Args:
        noise_scheduler: The scheduler to prepare
        device: Target device
        enable_zero_terminal_snr: Whether to fix betas for zero terminal SNR
    """
    # Move scheduler tensors to device
    if hasattr(noise_scheduler, 'alphas_cumprod'):
        noise_scheduler.alphas_cumprod = noise_scheduler.alphas_cumprod.to(device)
    
    if enable_zero_terminal_snr:
        fix_noise_scheduler_betas_for_zero_terminal_snr(noise_scheduler)


def apply_noise_offset(
    noise: torch.Tensor,
    noise_offset_type: str,
    noise_offset: float,
    noise_offset_random_strength: float = 0.0,
    adaptive_noise_scale: Optional[float] = None,
    latents: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Apply noise offset to improve dark/light training.
    
    Args:
        noise: Original noise tensor
        noise_offset: Base noise offset strength
        noise_offset_random_strength: Random variation in offset
        adaptive_noise_scale: Scale offset by latent std if provided
        latents: Latent tensor (required if adaptive_noise_scale is set)
    
    Returns:
        Modified noise tensor
    """
    if noise_offset <= 0.0 and noise_offset_random_strength <= 0.0:
        return noise

    mode = (noise_offset_type or "original").strip().lower()
    
    # Calculate actual offset
    if noise_offset_random_strength > 0.0:
        offset = noise_offset + torch.randn(1, device=noise.device).item() * noise_offset_random_strength
    else:
        offset = noise_offset
    
    # Apply adaptive scaling if provided
    if adaptive_noise_scale is not None and latents is not None:
        offset = offset * latents.std(dim=[1, 2, 3], keepdim=True)
    
    # Add offset noise
    if mode == "alternative":
        # Channel-wise full noise (stronger); kept optional via type switch.
        noise = noise + offset * torch.randn_like(noise)
    else:
        # Original (kohya-style): per-channel scalar offset.
        noise = noise + offset * torch.randn(
            noise.shape[0], noise.shape[1], 1, 1,
            device=noise.device, dtype=noise.dtype
        )
    
    return noise


def apply_input_perturbation(
    noise: torch.Tensor,
    ip_noise_gamma: float,
) -> torch.Tensor:
    """
    Apply input perturbation noise.
    
    From https://arxiv.org/abs/2301.11706
    Adds a small amount of random noise to the input noise for regularization.
    
    Args:
        noise: Original noise tensor
        ip_noise_gamma: Perturbation strength (0 = disabled)
    
    Returns:
        Perturbed noise tensor
    """
    if ip_noise_gamma <= 0.0:
        return noise
    
    # Add perturbation noise
    perturb_noise = torch.randn_like(noise)
    noise = noise + ip_noise_gamma * perturb_noise
    
    return noise


def get_timesteps_with_range(
    noise_scheduler: DDPMScheduler,
    batch_size: int,
    device: torch.device,
    min_timestep: Optional[int] = None,
    max_timestep: Optional[int] = None,
) -> torch.Tensor:
    """
    Sample random timesteps within the specified range.
    
    Args:
        noise_scheduler: The noise scheduler
        batch_size: Number of timesteps to sample
        device: Target device
        min_timestep: Minimum timestep (inclusive)
        max_timestep: Maximum timestep (inclusive)
    
    Returns:
        Tensor of sampled timesteps
    """
    num_train_timesteps = noise_scheduler.config.num_train_timesteps
    
    # Apply range limits
    min_ts = 0 if min_timestep is None else max(0, min_timestep)
    max_ts = (num_train_timesteps - 1) if max_timestep is None else min(num_train_timesteps - 1, max_timestep)
    
    # Validate range
    if min_ts > max_ts:
        min_ts, max_ts = 0, num_train_timesteps - 1
    
    # Sample timesteps
    timesteps = torch.randint(min_ts, max_ts + 1, (batch_size,), device=device).long()
    
    return timesteps
